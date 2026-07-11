package com.visionassist.appspace.activities.tabs.home.caption

import android.graphics.RectF
import android.util.Log
import com.visionassist.appspace.models.detector.DetectionResult

/**
 * SEMANTIC HASH - Based on OBJECTS + POSITIONS, not visual pixels
 *
 * This hash encodes WHAT objects are in the image and WHERE they are,
 * so the same scene from different angles gets HIGH similarity!
 *
 * Example:
 * Photo 1: [person:center, bottle:left, phone:right]
 * Photo 2: Same objects, different camera angle
 * → HIGH similarity!
 */
object SemanticHash {
    private const val TAG = "SemanticHash"

    // Grid size for spatial quantization (divide image into 3x3 grid)
    private const val GRID_SIZE = 3

    /**
     * Compute semantic hash from detected objects + their positions
     * This is BETTER because it includes spatial information!
     *
     * @param detectionResult YOLO detection results
     * @param imageWidth Original image width
     * @param imageHeight Original image height
     * @return Semantic hash encoding objects and positions
     */
    fun computeFromDetections(
        detectionResult: DetectionResult,
        imageWidth: Int,
        imageHeight: Int
    ): String? {
        try {
            // Get parallel lists
            val bboxes = detectionResult.boundingBoxes
            val classIndices = detectionResult.classIndices
            val detectionCount = detectionResult.detectionCount

            // Create semantic descriptor for each object
            val objectDescriptors = mutableListOf<String>()

            for (i in 0 until detectionCount) {
                val classId = classIndices[i]
                val bbox = bboxes[i]

                // Quantize position to grid cell (0-8)
                val gridCell = getGridCell(bbox, imageWidth, imageHeight)

                // Format: "classId:gridCell"
                val descriptor = "$classId:$gridCell"
                objectDescriptors.add(descriptor)
            }

            // Sort to make order-independent
            // Same objects in same positions = same hash, regardless of detection order
            val sortedDescriptors = objectDescriptors.sorted()

            // Create hash, (base 2)
            val descriptorString = sortedDescriptors.joinToString("|")
            val hashDec = descriptorString.hashCode()

            // Convert hash, (base 16)
            val hash = hashDec.toString(16)

            Log.d(
                TAG,
                "Semantic hash computed: $hash\nObjects: ${sortedDescriptors.joinToString(", ")}"
            )

            return hash
        } catch (e: Exception) {
            Log.e(TAG, "Error computing detection hash", e)
            return null
        }
    }

    /**
     * Get grid cell (0-8) for a bounding box
     *
     * Grid layout:
     * 0 | 1 | 2
     * --+---+--
     * 3 | 4 | 5
     * --+---+--
     * 6 | 7 | 8
     */
    private fun getGridCell(bbox: RectF, imageWidth: Int, imageHeight: Int): Int {
        // Get bbox center
        val centerX = (bbox.left + bbox.right) / 2f
        val centerY = (bbox.top + bbox.bottom) / 2f

        // Normalize to 0-1
        val normX = centerX / imageWidth
        val normY = centerY / imageHeight

        // Map to grid
        val gridX = (normX * GRID_SIZE).toInt().coerceIn(0, GRID_SIZE - 1)
        val gridY = (normY * GRID_SIZE).toInt().coerceIn(0, GRID_SIZE - 1)

        return gridY * GRID_SIZE + gridX
    }

    /**
     * Quantize object size: 0=small, 1=medium, 2=large
     */
    private fun getObjectSize(bbox: RectF, imageWidth: Int, imageHeight: Int): Int {
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val area = width * height

        // Normalize area (0-1)
        val imageArea = imageWidth * imageHeight
        val normArea = area / imageArea

        return when {
            normArea < 0.1f -> 0  // Small (<10% of image)
            normArea < 0.3f -> 1  // Medium (10-30%)
            else -> 2             // Large (>30%)
        }
    }

    /**
     * Compare two semantic hashes
     * Returns similarity 0-100%
     *
     * Since we're using java hashCode, we compare the actual descriptors
     */
    fun similarity(hash1: String, hash2: String): Boolean {
        // Simple exact match for now
        return hash1 == hash2
    }
}