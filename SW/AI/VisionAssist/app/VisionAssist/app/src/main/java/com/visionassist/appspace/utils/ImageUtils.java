package com.visionassist.appspace.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/**
 * Utility class for image conversion operations
 * Particularly for CameraX ImageProxy to Bitmap conversion
 */
public class ImageUtils {
    private static final String TAG = "ImageUtils";

    /**
     * Convert CameraX ImageProxy to Bitmap
     * Handles both JPEG and YUV formats
     *
     * @param imageProxy ImageProxy from CameraX
     * @return Bitmap or null if conversion fails
     */
    public static Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        try {
            int format = imageProxy.getFormat();

            if (format == ImageFormat.JPEG) {
                // JPEG format - direct conversion
                return jpegImageProxyToBitmap(imageProxy);
            } else if (format == ImageFormat.YUV_420_888) {
                // YUV format - needs conversion
                return yuvImageProxyToBitmap(imageProxy);
            } else {
                Log.e(TAG, "Unsupported image format: " + format);
                return null;
            }

        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e);
            return null;
        }
    }

    /**
     * Convert JPEG ImageProxy to Bitmap
     */
    private static Bitmap jpegImageProxyToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    /**
     * Convert YUV ImageProxy to Bitmap
     * Converts YUV_420_888 to RGB Bitmap
     */
    @OptIn(markerClass = ExperimentalGetImage.class)
    private static Bitmap yuvImageProxyToBitmap(ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image == null) {
            return null;
        }

        byte[] nv21 = yuv420888ToNv21(image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21,
                image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(
                new Rect(0, 0, image.getWidth(), image.getHeight()),
                100, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    /**
     * Convert YUV_420_888 to NV21 byte array
     */
    private static byte[] yuv420888ToNv21(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // Y channel
        yBuffer.get(nv21, 0, ySize);

        // VU interleaved (NV21 format)
        int uvIndex = ySize;
        for (int i = 0; i < vSize; i++) {
            nv21[uvIndex++] = vBuffer.get(i);
            nv21[uvIndex++] = uBuffer.get(i);
        }

        return nv21;
    }

    /**
     * Resize bitmap while maintaining aspect ratio
     */
    public static Bitmap resizeBitmap(Bitmap bitmap, int maxWidth, int maxHeight) {
        if (bitmap == null) return null;

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        float scale = Math.min(
                (float) maxWidth / width,
                (float) maxHeight / height
        );

        if (scale >= 1.0f) {
            return bitmap;
        }

        int newWidth = Math.round(width * scale);
        int newHeight = Math.round(height * scale);

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
    }

    /**
     * Crop bitmap to center square
     */
    public static Bitmap cropToSquare(Bitmap bitmap) {
        if (bitmap == null) return null;

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int size = Math.min(width, height);
        int x = (width - size) / 2;
        int y = (height - size) / 2;

        return Bitmap.createBitmap(bitmap, x, y, size, size);
    }
}