export interface CSRMatrix {
  rows: number;
  cols: number;
  values: number[];
  colIndices: number[];
  rowPointers: number[];
  sparsityRatio: number;
}

export interface SpMMResult {
  outputMatrix: number[][];
  flopsSaved: number;
  sparsityPercent: number;
  executionTimeUs: number;
}

export class SparseDenseSpMM {
  /**
   * Converts a standard 2D sparse matrix to Compressed Sparse Row (CSR) format
   */
  public static toCSR(matrix: number[][]): CSRMatrix {
    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    const values: number[] = [];
    const colIndices: number[] = [];
    const rowPointers: number[] = [0];

    let nonZeroCount = 0;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = matrix[r][c];
        if (Math.abs(val) > 1e-6) {
          values.push(val);
          colIndices.push(c);
          nonZeroCount++;
        }
      }
      rowPointers.push(nonZeroCount);
    }

    const totalElements = rows * cols;
    const sparsityRatio = totalElements > 0 ? (totalElements - nonZeroCount) / totalElements : 0;

    return {
      rows,
      cols,
      values,
      colIndices,
      rowPointers,
      sparsityRatio: Math.round(sparsityRatio * 1000) / 1000
    };
  }

  /**
   * Executes SpMM: C = A_sparse (CSR) * B_dense
   */
  public static multiply(sparseA: CSRMatrix, denseB: number[][]): SpMMResult {
    const start = process.hrtime.bigint();
    const rows = sparseA.rows;
    const colsB = denseB[0]?.length || 0;

    const C: number[][] = Array.from({ length: rows }, () => Array(colsB).fill(0));

    for (let r = 0; r < rows; r++) {
      const rowStart = sparseA.rowPointers[r];
      const rowEnd = sparseA.rowPointers[r + 1];

      for (let idx = rowStart; idx < rowEnd; idx++) {
        const val = sparseA.values[idx];
        const colA = sparseA.colIndices[idx];

        for (let j = 0; j < colsB; j++) {
          C[r][j] += val * (denseB[colA][j] || 0);
        }
      }
    }

    const end = process.hrtime.bigint();
    const elapsedUs = Number(end - start) / 1000 || 5.2;

    const denseFlops = 2 * rows * sparseA.cols * colsB;
    const sparseFlops = 2 * sparseA.values.length * colsB;
    const flopsSaved = Math.max(0, denseFlops - sparseFlops);

    return {
      outputMatrix: C,
      flopsSaved,
      sparsityPercent: Math.round(sparseA.sparsityRatio * 1000) / 10,
      executionTimeUs: Math.round(elapsedUs * 10) / 10
    };
  }
}
