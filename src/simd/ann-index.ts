import { SIMDVectorEngine } from './vector-wasm.js';

/**
 * Search result from ANN index
 */
export interface ANNSearchResult {
  id: string;
  distance: number;
  vector: Float32Array;
}

/**
 * IVF-Flat ANN Index — Approximate Nearest Neighbor search using
 * Inverted File Index with flat (exhaustive) scan within clusters.
 *
 * Algorithm:
 * ```
 *   Build:
 *     1. Run k-means to find C centroid vectors
 *     2. Assign each database vector to its nearest centroid (Voronoi cell)
 *
 *   Search:
 *     1. Find nprobe nearest centroids to query
 *     2. Exhaustively scan vectors within those clusters
 *     3. Return top-K nearest neighbors
 * ```
 *
 * Complexity:
 * - Build: O(n * C * d * iterations)
 * - Search: O(C * d + nprobe * n/C * d)
 *
 * Reference: Jégou et al., "Searching in one billion vectors" (2011)
 */
export class IVFFlatIndex {
  private centroids: Float32Array[] = [];
  private clusters: Map<number, Array<{ id: string; vector: Float32Array }>> = new Map();
  private numClusters: number;
  private dimension: number;

  constructor(numClusters: number = 8, dimension: number = 128) {
    this.numClusters = numClusters;
    this.dimension = dimension;
  }

  /**
   * Computes Euclidean distance between two vectors.
   */
  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  /**
   * Simplified k-means initialization using random sampling from data.
   */
  private initializeCentroids(data: Float32Array[]): void {
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    this.centroids = shuffled.slice(0, this.numClusters).map((v) => new Float32Array(v));
    for (let i = 0; i < this.numClusters; i++) {
      this.clusters.set(i, []);
    }
  }

  /**
   * Finds the nearest centroid index for a given vector.
   */
  private findNearestCentroid(vector: Float32Array): number {
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let c = 0; c < this.centroids.length; c++) {
      const dist = this.euclideanDistance(vector, this.centroids[c]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = c;
      }
    }
    return bestIdx;
  }

  /**
   * Builds the IVF-Flat index from a collection of named vectors.
   * Runs k-means for the specified number of iterations.
   */
  public build(data: Array<{ id: string; vector: Float32Array }>, kmeansIterations: number = 10): void {
    const vectors = data.map((d) => d.vector);
    this.initializeCentroids(vectors);

    for (let iter = 0; iter < kmeansIterations; iter++) {
      // Clear clusters
      for (let c = 0; c < this.numClusters; c++) {
        this.clusters.set(c, []);
      }

      // Assign vectors to nearest centroid
      for (const item of data) {
        const c = this.findNearestCentroid(item.vector);
        this.clusters.get(c)!.push(item);
      }

      // Update centroids
      for (let c = 0; c < this.numClusters; c++) {
        const members = this.clusters.get(c)!;
        if (members.length === 0) continue;

        const newCentroid = new Float32Array(this.dimension);
        for (const m of members) {
          for (let d = 0; d < this.dimension; d++) {
            newCentroid[d] += m.vector[d];
          }
        }
        for (let d = 0; d < this.dimension; d++) {
          newCentroid[d] /= members.length;
        }
        this.centroids[c] = newCentroid;
      }
    }
  }

  /**
   * Searches for the K nearest neighbors of a query vector.
   * Probes `nprobe` nearest clusters for approximate results.
   */
  public search(query: Float32Array, topK: number = 5, nprobe: number = 2): ANNSearchResult[] {
    // Find nprobe nearest centroids
    const centroidDistances = this.centroids.map((c, i) => ({
      idx: i,
      dist: this.euclideanDistance(query, c),
    }));
    centroidDistances.sort((a, b) => a.dist - b.dist);
    const probeClusters = centroidDistances.slice(0, nprobe);

    // Exhaustive scan within probed clusters
    const candidates: ANNSearchResult[] = [];
    for (const { idx } of probeClusters) {
      const members = this.clusters.get(idx) || [];
      for (const m of members) {
        candidates.push({
          id: m.id,
          distance: this.euclideanDistance(query, m.vector),
          vector: m.vector,
        });
      }
    }

    // Sort by distance and return top-K
    candidates.sort((a, b) => a.distance - b.distance);
    return candidates.slice(0, topK);
  }

  /**
   * Returns index statistics.
   */
  public getStats(): { numClusters: number; totalVectors: number; clusterSizes: number[] } {
    const clusterSizes: number[] = [];
    let totalVectors = 0;
    for (let c = 0; c < this.numClusters; c++) {
      const size = this.clusters.get(c)?.length || 0;
      clusterSizes.push(size);
      totalVectors += size;
    }
    return { numClusters: this.numClusters, totalVectors, clusterSizes };
  }
}
