package com.visionassist.appspace.activities.tabs.home.caption

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Perceptual Hash (pHash) implementation for image similarity detection
 * Based on: https://github.com/pragone/jphash
 *
 * This implementation uses DCT (Discrete Cosine Transform) to create a
 * perceptual hash that is robust to minor image changes.
 */
object PerceptualHash {
    private const val TAG = "PerceptualHash"

    // Standard pHash parameters
    private const val HASH_SIZE = 8  // Hash will be 8x8 = 64 bits
    private const val SMALL_SIZE = 32  // Resize to 32x32 before DCT

    /**
     * Compute perceptual hash of a bitmap
     * Returns a 64-bit hash as hex string (16 characters)
     */
    fun computeHash(bitmap: Bitmap): String {
        try {
            // Step 1: Resize to 32x32
            val resized = Bitmap.createScaledBitmap(bitmap, SMALL_SIZE, SMALL_SIZE, false)

            // Step 2: Convert to grayscale
            val grayscale = Array(SMALL_SIZE) { DoubleArray(SMALL_SIZE) }
            for (y in 0 until SMALL_SIZE) {
                for (x in 0 until SMALL_SIZE) {
                    val pixel = resized.getPixel(x, y)
                    val r = Color.red(pixel)
                    val g = Color.green(pixel)
                    val b = Color.blue(pixel)
                    // Standard grayscale conversion
                    grayscale[y][x] = (r * 0.299 + g * 0.587 + b * 0.114)
                }
            }

            resized.recycle()

            // Step 3: Compute DCT (Discrete Cosine Transform)
            val dct = computeDCT(grayscale)

            // Step 4: Extract top-left 8x8 DCT values (low frequencies)
            val dctLowFreq = Array(HASH_SIZE) { DoubleArray(HASH_SIZE) }
            for (y in 0 until HASH_SIZE) {
                for (x in 0 until HASH_SIZE) {
                    dctLowFreq[y][x] = dct[y][x]
                }
            }

            // Step 5: Compute median of DCT values
            val values = mutableListOf<Double>()
            for (y in 0 until HASH_SIZE) {
                for (x in 0 until HASH_SIZE) {
                    values.add(dctLowFreq[y][x])
                }
            }
            values.sort()
            val median = values[values.size / 2]

            // Step 6: Generate hash (1 if > median, 0 otherwise)
            var hash = 0L
            for (y in 0 until HASH_SIZE) {
                for (x in 0 until HASH_SIZE) {
                    if (dctLowFreq[y][x] > median) {
                        hash = hash or (1L shl (y * HASH_SIZE + x))
                    }
                }
            }

            // Convert to hex string (16 characters for 64 bits)
            return hash.toString(16).padStart(16, '0')

        } catch (e: Exception) {
            Log.e(TAG, "Error computing perceptual hash", e)
            // Fallback: return timestamp-based unique hash
            return System.currentTimeMillis().toString(16).padStart(16, '0')
        }
    }

    /**
     * Compute 2D Discrete Cosine Transform (DCT)
     * This is the core of the perceptual hash algorithm
     */
    private fun computeDCT(input: Array<DoubleArray>): Array<DoubleArray> {
        val N = input.size
        val output = Array(N) { DoubleArray(N) }

        for (u in 0 until N) {
            for (v in 0 until N) {
                var sum = 0.0

                for (i in 0 until N) {
                    for (j in 0 until N) {
                        sum += input[i][j] *
                                cos((2 * i + 1) * u * Math.PI / (2.0 * N)) *
                                cos((2 * j + 1) * v * Math.PI / (2.0 * N))
                    }
                }

                // Apply DCT normalization
                val cu = if (u == 0) 1.0 / sqrt(2.0) else 1.0
                val cv = if (v == 0) 1.0 / sqrt(2.0) else 1.0

                output[u][v] = 0.25 * cu * cv * sum
            }
        }

        return output
    }

    /**
     * Compute Hamming distance between two hash strings
     * Returns number of different bits (0-64)
     */
    fun hammingDistance(hash1: String, hash2: String): Int {
        if (hash1.length != hash2.length) return 64 // Max distance

        try {
            val long1 = hash1.toLong(16)
            val long2 = hash2.toLong(16)
            var xor = long1 xor long2

            // Count number of 1 bits (Hamming distance)
            var distance = 0
            while (xor != 0L) {
                distance += (xor and 1L).toInt()
                xor = xor ushr 1
            }

            return distance

        } catch (e: Exception) {
            Log.e(TAG, "Error computing Hamming distance", e)
            return 64 // Max distance on error
        }
    }

    /**
     * Compute similarity percentage between two hashes
     * Returns 0-100, where 100 = identical
     */
    fun similarity(hash1: String, hash2: String): Float {
        val distance = hammingDistance(hash1, hash2)
        return (1f - distance / 64f) * 100f
    }

    /**
     * Check if two hashes are similar (above threshold)
     * Default threshold: 90% similarity
     */
    fun isSimilar(hash1: String, hash2: String, threshold: Float = 90f): Boolean {
        return similarity(hash1, hash2) >= threshold
    }
}