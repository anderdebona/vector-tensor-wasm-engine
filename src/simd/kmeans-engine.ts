import { CosineSimilarityWasmSIMD } from './cosine-simd.js';

export interface KMeansResult {
  centroids: Float32Array[];
  assignments: number[];
  inertia: number;
  iterations: number;
}

export class KMeansClusteringEngine {
  private k: number;
  private maxIterations: number;
  private tolerance: number;

  constructor(k: number = 4, maxIterations: number = 20, tolerance: number = 1e-4) {
    this.k = k;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  public fit(vectors: Float32Array[]): KMeansResult {
    if (vectors.length < this.k) {
      throw new Error(`Insufficient vectors (${vectors.length}) for k=${this.k}`);
    }

    const dim = vectors[0].length;
    // Initialize centroids using k-means++ style sampling
    let centroids: Float32Array[] = [new Float32Array(vectors[0])];
    while (centroids.length < this.k) {
      const idx = Math.floor((centroids.length / this.k) * vectors.length) % vectors.length;
      centroids.push(new Float32Array(vectors[idx]));
    }

    let assignments = new Array<number>(vectors.length).fill(0);
    let prevInertia = Infinity;
    let iteration = 0;

    for (iteration = 0; iteration < this.maxIterations; iteration++) {
      let currentInertia = 0.0;

      // Assign each vector to closest centroid (Euclidean distance)
      for (let i = 0; i < vectors.length; i++) {
        let bestDist = Infinity;
        let bestCluster = 0;

        for (let c = 0; c < this.k; c++) {
          let dist = 0.0;
          for (let d = 0; d < dim; d++) {
            const diff = vectors[i][d] - centroids[c][d];
            dist += diff * diff;
          }
          if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
          }
        }

        assignments[i] = bestCluster;
        currentInertia += bestDist;
      }

      // Recompute centroids
      const counts = new Array<number>(this.k).fill(0);
      const newCentroids = Array.from({ length: this.k }, () => new Float32Array(dim));

      for (let i = 0; i < vectors.length; i++) {
        const cluster = assignments[i];
        counts[cluster]++;
        for (let d = 0; d < dim; d++) {
          newCentroids[cluster][d] += vectors[i][d];
        }
      }

      for (let c = 0; c < this.k; c++) {
        if (counts[c] > 0) {
          for (let d = 0; d < dim; d++) {
            newCentroids[c][d] /= counts[c];
          }
        } else {
          newCentroids[c].set(centroids[c]);
        }
      }

      centroids = newCentroids;

      if (Math.abs(prevInertia - currentInertia) < this.tolerance) {
        break;
      }
      prevInertia = currentInertia;
    }

    return {
      centroids,
      assignments,
      inertia: prevInertia === Infinity ? 0 : prevInertia,
      iterations: iteration + 1,
    };
  }
}
