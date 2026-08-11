export class TensorMatMul {
  public static multiply(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
    const result = new Float32Array(M * N);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
        result[i * N + j] = sum;
      }
    }
    return result;
  }
  public static transpose(matrix: Float32Array, rows: number, cols: number): Float32Array {
    const result = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) result[j * rows + i] = matrix[i * cols + j];
    return result;
  }
  public static frobenius(matrix: Float32Array): number {
    let sum = 0; for (let i = 0; i < matrix.length; i++) sum += matrix[i] ** 2; return Math.sqrt(sum);
  }
}
