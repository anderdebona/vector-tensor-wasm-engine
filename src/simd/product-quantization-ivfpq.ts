export interface PQCodebook {
  numSubspaces: number; // M
  subvectorDim: number; // D / M
  centroids: number[][][]; // [subspace][centroidIdx][subvectorDim]
}

export interface EncodedPQVector {
  id: string;
  codes: number[]; // length M, containing centroid indices
}

export class ProductQuantizationIVFPQ {
  private codebook: PQCodebook;

  constructor(totalDim: number = 8, numSubspaces: number = 2, centroidsPerSubspace: number = 4) {
    const subvectorDim = Math.floor(totalDim / numSubspaces);
    const centroids: number[][][] = [];

    // Initialize mock centroid codebooks
    for (let m = 0; m < numSubspaces; m++) {
      centroids[m] = [];
      for (let k = 0; k < centroidsPerSubspace; k++) {
        const centroid = Array.from({ length: subvectorDim }, (_, i) => (k + 1) * 0.25 + i * 0.1);
        centroids[m].push(centroid);
      }
    }

    this.codebook = {
      numSubspaces,
      subvectorDim,
      centroids
    };
  }

  /**
   * Encodes a high-dimensional vector into M compact 8-bit codebook indices
   */
  public encode(id: string, vector: number[]): EncodedPQVector {
    const codes: number[] = [];

    for (let m = 0; m < this.codebook.numSubspaces; m++) {
      const subvec = vector.slice(m * this.codebook.subvectorDim, (m + 1) * this.codebook.subvectorDim);
      let bestDist = Infinity;
      let bestIdx = 0;

      this.codebook.centroids[m].forEach((centroid, idx) => {
        let dist = 0;
        for (let i = 0; i < subvec.length; i++) {
          const diff = (subvec[i] || 0) - centroid[i];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      });

      codes.push(bestIdx);
    }

    return { id, codes };
  }

  /**
   * Performs Asymmetric Distance Computation (ADC) between uncompressed query q and PQ code
   */
  public computeADCDistance(query: number[], encoded: EncodedPQVector): number {
    let totalDist = 0;

    for (let m = 0; m < this.codebook.numSubspaces; m++) {
      const querySubvec = query.slice(m * this.codebook.subvectorDim, (m + 1) * this.codebook.subvectorDim);
      const centroid = this.codebook.centroids[m][encoded.codes[m]];

      if (centroid) {
        for (let i = 0; i < querySubvec.length; i++) {
          const diff = (querySubvec[i] || 0) - centroid[i];
          totalDist += diff * diff;
        }
      }
    }

    return Math.round(Math.sqrt(totalDist) * 1000) / 1000;
  }

  public getCodebookStats() {
    return {
      numSubspaces: this.codebook.numSubspaces,
      subvectorDim: this.codebook.subvectorDim,
      centroidsPerSubspace: this.codebook.centroids[0]?.length || 0,
      compressionRatio: `${(this.codebook.numSubspaces * this.codebook.subvectorDim * 4) / this.codebook.numSubspaces}x bytes saved`
    };
  }
}
