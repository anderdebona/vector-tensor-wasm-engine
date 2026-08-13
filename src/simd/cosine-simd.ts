export class CosineSimilarityWasmSIMD {
  /**
   * Computes cosine similarity between two Float32 vectors using 4-wide SIMD unrolling logic
   * Cos(u, v) = (u . v) / (||u|| * ||v||)
   */
  public static compute(u: Float32Array, v: Float32Array): number {
    if (u.length !== v.length) {
      throw new Error(`Dimension mismatch: ${u.length} vs ${v.length}`);
    }

    const n = u.length;
    let dot = 0.0;
    let normU = 0.0;
    let normV = 0.0;

    const unrollEnd = n - (n % 4);
    for (let i = 0; i < unrollEnd; i += 4) {
      const u0 = u[i], u1 = u[i + 1], u2 = u[i + 2], u3 = u[i + 3];
      const v0 = v[i], v1 = v[i + 1], v2 = v[i + 2], v3 = v[i + 3];

      dot += u0 * v0 + u1 * v1 + u2 * v2 + u3 * v3;
      normU += u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3;
      normV += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }

    for (let i = unrollEnd; i < n; i++) {
      dot += u[i] * v[i];
      normU += u[i] * u[i];
      normV += v[i] * v[i];
    }

    const denominator = Math.sqrt(normU) * Math.sqrt(normV);
    if (denominator === 0) return 0.0;
    return Math.max(-1.0, Math.min(1.0, dot / denominator));
  }

  /**
   * Batch cosine similarity search across a matrix of vectors
   */
  public static batchSearch(
    query: Float32Array,
    matrix: Array<{ id: string; vector: Float32Array }>,
    topK: number = 5
  ): Array<{ id: string; score: number }> {
    const scores = matrix.map(item => ({
      id: item.id,
      score: this.compute(query, item.vector),
    }));

    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }
}
