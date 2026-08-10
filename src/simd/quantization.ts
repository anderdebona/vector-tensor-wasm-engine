/**
 * Quantization method enum
 */
export type QuantizationMethod = 'SCALAR' | 'PRODUCT';

/**
 * Result of vector quantization including compression ratio
 */
export interface QuantizationResult {
  method: QuantizationMethod;
  originalDimension: number;
  quantizedValues: Uint8Array;
  scale: number;
  offset: number;
  compressionRatio: number;
  reconstructionError: number;
}

/**
 * Vector Quantization Engine — Compresses high-dimensional float embeddings
 * into compact integer representations for efficient storage and search.
 *
 * Implements two quantization strategies:
 *
 * 1. **Scalar Quantization (SQ8)**: Maps each float to 8-bit uint
 *    ```
 *    q = round((x - min) / (max - min) * 255)
 *    x_hat = q / 255 * (max - min) + min
 *    ```
 *
 * 2. **Product Quantization (PQ)**: Splits vector into M subspaces,
 *    quantizes each independently for finer granularity.
 *    ```
 *    v = [v1 | v2 | ... | vM]  →  [q1, q2, ..., qM]
 *    ```
 *
 * Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search"
 * (IEEE TPAMI, 2011)
 */
export class VectorQuantizer {
  /**
   * Scalar Quantization — Maps Float32 vector to Uint8 with linear scaling.
   * Achieves 4x compression (float32 → uint8).
   */
  public static scalarQuantize(vector: Float32Array): QuantizationResult {
    const min = vector.reduce((a, b) => Math.min(a, b), Infinity);
    const max = vector.reduce((a, b) => Math.max(a, b), -Infinity);
    const range = max - min || 1;

    const quantized = new Uint8Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      quantized[i] = Math.round(((vector[i] - min) / range) * 255);
    }

    // Compute reconstruction error
    let errorSum = 0;
    for (let i = 0; i < vector.length; i++) {
      const reconstructed = (quantized[i] / 255) * range + min;
      errorSum += (vector[i] - reconstructed) ** 2;
    }
    const reconstructionError = Math.sqrt(errorSum / vector.length);

    return {
      method: 'SCALAR',
      originalDimension: vector.length,
      quantizedValues: quantized,
      scale: range,
      offset: min,
      compressionRatio: 4.0, // float32 (4 bytes) → uint8 (1 byte)
      reconstructionError,
    };
  }

  /**
   * Dequantizes a scalar-quantized vector back to Float32.
   */
  public static scalarDequantize(result: QuantizationResult): Float32Array {
    const output = new Float32Array(result.originalDimension);
    for (let i = 0; i < result.originalDimension; i++) {
      output[i] = (result.quantizedValues[i] / 255) * result.scale + result.offset;
    }
    return output;
  }

  /**
   * Product Quantization — Splits vector into M subspaces and quantizes
   * each independently.
   */
  public static productQuantize(vector: Float32Array, numSubspaces: number = 4): QuantizationResult {
    const subDim = Math.floor(vector.length / numSubspaces);
    const quantized = new Uint8Array(vector.length);
    let totalError = 0;

    for (let m = 0; m < numSubspaces; m++) {
      const start = m * subDim;
      const end = m === numSubspaces - 1 ? vector.length : start + subDim;
      const subVector = vector.slice(start, end);

      const min = subVector.reduce((a, b) => Math.min(a, b), Infinity);
      const max = subVector.reduce((a, b) => Math.max(a, b), -Infinity);
      const range = max - min || 1;

      for (let i = start; i < end; i++) {
        quantized[i] = Math.round(((vector[i] - min) / range) * 255);
        const reconstructed = (quantized[i] / 255) * range + min;
        totalError += (vector[i] - reconstructed) ** 2;
      }
    }

    return {
      method: 'PRODUCT',
      originalDimension: vector.length,
      quantizedValues: quantized,
      scale: 0,
      offset: 0,
      compressionRatio: 4.0,
      reconstructionError: Math.sqrt(totalError / vector.length),
    };
  }
}
