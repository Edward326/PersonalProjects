@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.newprofile

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.graphics.toColorInt
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_infoBB
import com.visionassist.appspace.utils.load_infoCaption
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate
import kotlin.math.roundToInt

class UserAccessibility1Activity : ComponentActivity() {
    private val TAG = "UserAccessibility1Activity"

    // Current section (1 = BoundingBox, 2 = Caption)
    private val currentSection = mutableIntStateOf(1)

    // BoundingBox section states
    private val bboxRedValue = mutableFloatStateOf(0f)
    private val bboxGreenValue = mutableFloatStateOf(255f / 2.55f)
    private val bboxBlueValue = mutableFloatStateOf(0f)

    private val textRedValue = mutableFloatStateOf(255f / 2.55f)
    private val textGreenValue = mutableFloatStateOf(255f / 2.55f)
    private val textBlueValue = mutableFloatStateOf(255f / 2.55f)

    private val bgRedValue = mutableFloatStateOf(0f)
    private val bgGreenValue = mutableFloatStateOf(0f)
    private val bgBlueValue = mutableFloatStateOf(0f)

    private val isBold = mutableStateOf(true)
    private val showConfidence = mutableStateOf(true)

    // Caption section states
    private val captionTextRedValue = mutableFloatStateOf(90f)
    private val captionTextGreenValue = mutableFloatStateOf(90f)
    private val captionTextBlueValue = mutableFloatStateOf(90f)

    private val captionBgRedValue = mutableFloatStateOf(103f / 2.55f)
    private val captionBgGreenValue = mutableFloatStateOf(80f / 2.55f)
    private val captionBgBlueValue = mutableFloatStateOf(164f / 2.55f)

    private val hasHaptics = mutableStateOf(true)

    // Preview image state
    private val previewBitmap = mutableStateOf<Bitmap?>(null)

    // Update handler with debounce
    private val updateHandler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null

    // Info dialog manager
    private lateinit var infoNotificationManager: InfoNotificationManager
    private var infoDialogStep = 0

    @SuppressLint("ServiceCast")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        currentSection.intValue = intent.getIntExtra(Constants.EXTRA_USERACC_OPTION, 1)

        loadInitialValuesFromAppConfig()

        // Load initial preview
        updateBoundingBoxPreview()

        setContent {
            UserAccessibility1Screen(
                currentSection = currentSection.intValue,
                // BoundingBox states
                bboxRedValue = bboxRedValue.floatValue,
                bboxGreenValue = bboxGreenValue.floatValue,
                bboxBlueValue = bboxBlueValue.floatValue,
                textRedValue = textRedValue.floatValue,
                textGreenValue = textGreenValue.floatValue,
                textBlueValue = textBlueValue.floatValue,
                bgRedValue = bgRedValue.floatValue,
                bgGreenValue = bgGreenValue.floatValue,
                bgBlueValue = bgBlueValue.floatValue,
                isBold = isBold.value,
                showConfidence = showConfidence.value,
                // Caption states
                captionTextRedValue = captionTextRedValue.floatValue,
                captionTextGreenValue = captionTextGreenValue.floatValue,
                captionTextBlueValue = captionTextBlueValue.floatValue,
                captionBgRedValue = captionBgRedValue.floatValue,
                captionBgGreenValue = captionBgGreenValue.floatValue,
                captionBgBlueValue = captionBgBlueValue.floatValue,
                hasHaptics = hasHaptics.value,
                previewBitmap = previewBitmap.value,
                // Handlers
                onBboxRedChange = { handleBboxRedChange(it) },
                onBboxGreenChange = { handleBboxGreenChange(it) },
                onBboxBlueChange = { handleBboxBlueChange(it) },
                onTextRedChange = { handleTextRedChange(it) },
                onTextGreenChange = { handleTextGreenChange(it) },
                onTextBlueChange = { handleTextBlueChange(it) },
                onBgRedChange = { handleBgRedChange(it) },
                onBgGreenChange = { handleBgGreenChange(it) },
                onBgBlueChange = { handleBgBlueChange(it) },
                onBoldChange = { isBold.value = it; schedulePreviewUpdate(0) },
                onShowConfidenceChange = { showConfidence.value = it; schedulePreviewUpdate(0) },
                onCaptionTextRedChange = { handleCaptionTextRedChange(it) },
                onCaptionTextGreenChange = { handleCaptionTextGreenChange(it) },
                onCaptionTextBlueChange = { handleCaptionTextBlueChange(it) },
                onCaptionBgRedChange = { handleCaptionBgRedChange(it) },
                onCaptionBgGreenChange = { handleCaptionBgGreenChange(it) },
                onCaptionBgBlueChange = { handleCaptionBgBlueChange(it) },
                onHapticsChange = {
                    hasHaptics.value = it
                    if (it) {
                        vibrate(haptic_model0())
                    }
                    },
                    onInfoClick = ::handleInfoClick,
                    onBackClick = ::handleBackClick,
                    onNextClick = ::handleNextClick
                    )
                }
        }

        private fun loadInitialValuesFromAppConfig() {
            // Load BBox color if exists
            if (AppConfig.bbox_color != null) {
                val color1 = AppConfig.bbox_color.toColorInt()
                bboxRedValue.floatValue = android.graphics.Color.red(color1) / 2.55f
                bboxGreenValue.floatValue = android.graphics.Color.green(color1) / 2.55f
                bboxBlueValue.floatValue = android.graphics.Color.blue(color1) / 2.55f

                val color2 = AppConfig.label_color.toColorInt()
                textRedValue.floatValue = android.graphics.Color.red(color2) / 2.55f
                textGreenValue.floatValue = android.graphics.Color.green(color2) / 2.55f
                textBlueValue.floatValue = android.graphics.Color.blue(color2) / 2.55f

                val color3 = AppConfig.label_bck_color.toColorInt()
                bgRedValue.floatValue = android.graphics.Color.red(color3) / 2.55f
                bgGreenValue.floatValue = android.graphics.Color.green(color3) / 2.55f
                bgBlueValue.floatValue = android.graphics.Color.blue(color3) / 2.55f

                // Load Bold setting (default FALSE)
                isBold.value = AppConfig.isBold

                // Load Show Confidence setting (default FALSE)
                showConfidence.value = AppConfig.show_confidence
            }

            // Load Caption text color if exists
            if (AppConfig.caption_color != null) {
                val color1 = AppConfig.caption_color.toColorInt()
                captionTextRedValue.floatValue = android.graphics.Color.red(color1) / 2.55f
                captionTextGreenValue.floatValue = android.graphics.Color.green(color1) / 2.55f
                captionTextBlueValue.floatValue = android.graphics.Color.blue(color1) / 2.55f

                val color2 = AppConfig.caption_bck_color.toColorInt()
                captionBgRedValue.floatValue = android.graphics.Color.red(color2) / 2.55f
                captionBgGreenValue.floatValue = android.graphics.Color.green(color2) / 2.55f
                captionBgBlueValue.floatValue = android.graphics.Color.blue(color2) / 2.55f

                // Load Haptics setting (default TRUE)
                hasHaptics.value = AppConfig.haptics
            }
        }

        // BoundingBox color change handlers with debounce
        private fun handleBboxRedChange(value: Float) {
            bboxRedValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleBboxGreenChange(value: Float) {
            bboxGreenValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleBboxBlueChange(value: Float) {
            bboxBlueValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleTextRedChange(value: Float) {
            textRedValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleTextGreenChange(value: Float) {
            textGreenValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleTextBlueChange(value: Float) {
            textBlueValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleBgRedChange(value: Float) {
            bgRedValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleBgGreenChange(value: Float) {
            bgGreenValue.floatValue = value
            schedulePreviewUpdate()
        }

        private fun handleBgBlueChange(value: Float) {
            bgBlueValue.floatValue = value
            schedulePreviewUpdate()
        }

        // Caption color change handlers
        private fun handleCaptionTextRedChange(value: Float) {
            captionTextRedValue.floatValue = value
        }

        private fun handleCaptionTextGreenChange(value: Float) {
            captionTextGreenValue.floatValue = value
        }

        private fun handleCaptionTextBlueChange(value: Float) {
            captionTextBlueValue.floatValue = value
        }

        private fun handleCaptionBgRedChange(value: Float) {
            captionBgRedValue.floatValue = value
        }

        private fun handleCaptionBgGreenChange(value: Float) {
            captionBgGreenValue.floatValue = value
        }

        private fun handleCaptionBgBlueChange(value: Float) {
            captionBgBlueValue.floatValue = value
        }

        private fun schedulePreviewUpdate(delayMs: Long = Constants.PREVIEW_UPDATE_DELAY.toLong()) {
            // Cancel previous update
            updateRunnable?.let { updateHandler.removeCallbacks(it) }

            // Schedule new update
            updateRunnable = Runnable {
                updateBoundingBoxPreview()
            }
            updateHandler.postDelayed(updateRunnable!!, delayMs)
        }

        private fun updateBoundingBoxPreview() {
            val bitmap = BitmapFactory.decodeResource(resources, R.drawable.detection_preview)
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)

            // Get actual image dimensions
            val imageWidth = bitmap.width.toFloat()
            val imageHeight = bitmap.height.toFloat()

            // Convert 0-100 values to 0-255
            val bboxColor = android.graphics.Color.rgb(
                (bboxRedValue.floatValue * 2.55f).roundToInt(),
                (bboxGreenValue.floatValue * 2.55f).roundToInt(),
                (bboxBlueValue.floatValue * 2.55f).roundToInt()
            )

            val textColor = android.graphics.Color.rgb(
                (textRedValue.floatValue * 2.55f).roundToInt(),
                (textGreenValue.floatValue * 2.55f).roundToInt(),
                (textBlueValue.floatValue * 2.55f).roundToInt()
            )

            val bgColor = android.graphics.Color.rgb(
                (bgRedValue.floatValue * 2.55f).roundToInt(),
                (bgGreenValue.floatValue * 2.55f).roundToInt(),
                (bgBlueValue.floatValue * 2.55f).roundToInt()
            )

            // Get screen width for dynamic sizing
            val screenWidth = resources.displayMetrics.widthPixels.toFloat()

            // Draw ONLY 2 bounding boxes: Banana and Apple
            // Banana - on the counter
            drawBoundingBox(
                canvas,
                RectF(
                    imageWidth * 0.76f,  // Right side
                    imageHeight * 0.575f,
                    imageWidth * 0.97f,
                    imageHeight * 0.95f
                ),
                if (AppConfig.mainLanguage.code == "en") "banana" else "banană",
                89,
                bboxColor,
                textColor,
                bgColor,
                screenWidth
            )

            // Apple - in the bowl
            drawBoundingBox(
                canvas,
                RectF(
                    imageWidth * 0.20f,  // Center-left in bowl
                    imageHeight * 0.52f,
                    imageWidth * 0.43f,
                    imageHeight * 0.80f
                ),
                if (AppConfig.mainLanguage.code == "en") "apple" else "măr",
                92,
                bboxColor,
                textColor,
                bgColor,
                screenWidth
            )

            drawBoundingBox(
                canvas,
                RectF(
                    imageWidth * 0.05f,  // Left edge of orange
                    imageHeight * 0.60f,  // Top edge of orange
                    imageWidth * 0.25f,   // Right edge of orange
                    imageHeight * 0.87f   // Bottom edge of orange
                ),
                if (AppConfig.mainLanguage.code == "en") "orange" else "portocală",
                92,
                bboxColor,
                textColor,
                bgColor,
                screenWidth
            )

            previewBitmap.value = mutableBitmap
        }

        private fun drawBoundingBox(
            canvas: Canvas,
            rect: RectF,
            label: String,
            confidence: Int,
            bboxColor: Int,
            textColor: Int,
            bgColor: Int,
            screenWidth: Float
        ) {
            val strokeWidth = screenWidth * 0.02f

            // Draw bounding box with dynamic stroke width
            val paint = Paint().apply {
                color = bboxColor
                style = Paint.Style.STROKE
                this.strokeWidth = strokeWidth
            }
            canvas.drawRect(rect, paint)

            // Calculate text size based on screen width
            val textSize = screenWidth * 0.06f

            // Draw label text
            val textPaint = Paint().apply {
                color = textColor
                this.textSize = textSize
                isFakeBoldText = isBold.value
                isAntiAlias = true
            }

            val labelText = if (showConfidence.value) "$label $confidence%" else label
            val textBounds = android.graphics.Rect()
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)

            // Draw background with extra padding to fit text properly
            val bgPaint = Paint().apply {
                color = bgColor
                style = Paint.Style.FILL
            }

            // Widen background to fit text properly (add 16dp padding)
            val paddingHorizontal = 16f
            val paddingVertical = 8f

            val textBgRect = RectF(
                rect.left,
                rect.top - textBounds.height() - paddingVertical * 2,
                rect.left + textBounds.width() + paddingHorizontal * 2,  // ← WIDENED
                rect.top
            )
            canvas.drawRect(textBgRect, bgPaint)

            // Draw text centered in background
            canvas.drawText(
                labelText,
                rect.left + paddingHorizontal,
                rect.top - paddingVertical,
                textPaint
            )
        }

        private fun handleInfoClick() {
            if (currentSection.intValue == 1) {
                // BoundingBox section info
                val message = load_infoBB(this)

                infoNotificationManager.showNotification(
                    message,
                    { infoNotificationManager.hideNotification() },
                    "OK"
                )
            } else {
                // Caption section info - two-step dialog
                if (infoDialogStep == 0) {
                    val message = load_infoCaption(this, false)

                    infoNotificationManager.showNotification(
                        message,
                        {
                            infoDialogStep = 1
                            handleInfoClick()
                        },
                        if (AppConfig.mainLanguage.code == "en") "Next" else "Următorul"
                    )
                } else {
                    val message = load_infoCaption(this, true)

                    infoNotificationManager.showNotification(
                        message,
                        {
                            infoDialogStep = 0
                            infoNotificationManager.hideNotification()
                        },
                        "OK"
                    )
                }
            }
        }

        private fun handleBackClick() {
            if (currentSection.intValue == 2) {
                // From Caption section, go back to BoundingBox section
                ProfileFileCollection.deleteUserAccessibility1Activity(false)
                currentSection.intValue = 1
            } else {
                // From BoundingBox section, go back to previous activity
                if (AppConfig.isContributor) {
                    ProfileFileCollection.deleteUserInfoActivity(2)
                    val intent = Intent(this, UserInfoActivity::class.java)
                    intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 3)
                    startActivity(intent)
                    finish()
                } else {
                    ProfileFileCollection.deleteUserInfoActivity(0)
                    val intent = Intent(this, UserInfoActivity::class.java)
                    intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 1)
                    startActivity(intent)
                    finish()
                }
            }
        }

        private fun handleNextClick() {
            if (currentSection.intValue == 1) {
                // Save BoundingBox section
                val bboxColorHex = String.format(
                    "#%02X%02X%02X",
                    (bboxRedValue.floatValue * 2.55f).roundToInt(),
                    (bboxGreenValue.floatValue * 2.55f).roundToInt(),
                    (bboxBlueValue.floatValue * 2.55f).roundToInt()
                )

                val textColorHex = String.format(
                    "#%02X%02X%02X",
                    (textRedValue.floatValue * 2.55f).roundToInt(),
                    (textGreenValue.floatValue * 2.55f).roundToInt(),
                    (textBlueValue.floatValue * 2.55f).roundToInt()
                )

                val bgColorHex = String.format(
                    "#%02X%02X%02X",
                    (bgRedValue.floatValue * 2.55f).roundToInt(),
                    (bgGreenValue.floatValue * 2.55f).roundToInt(),
                    (bgBlueValue.floatValue * 2.55f).roundToInt()
                )

                ProfileFileCollection.writeUserAccessibility1BoundingBox(
                    bboxColorHex,
                    textColorHex,
                    bgColorHex,
                    isBold.value,
                    showConfidence.value
                )

                // Update AppConfig
                AppConfig.bbox_color = bboxColorHex
                AppConfig.label_color = textColorHex
                AppConfig.label_bck_color = bgColorHex
                AppConfig.isBold = isBold.value
                AppConfig.show_confidence = showConfidence.value

                // Move to Caption section
                currentSection.intValue = 2
            } else {
                // Save Caption section
                val captionTextColorHex = String.format(
                    "#%02X%02X%02X",
                    (captionTextRedValue.floatValue * 2.55f).roundToInt(),
                    (captionTextGreenValue.floatValue * 2.55f).roundToInt(),
                    (captionTextBlueValue.floatValue * 2.55f).roundToInt()
                )

                val captionBgColorHex = String.format(
                    "#%02X%02X%02X",
                    (captionBgRedValue.floatValue * 2.55f).roundToInt(),
                    (captionBgGreenValue.floatValue * 2.55f).roundToInt(),
                    (captionBgBlueValue.floatValue * 2.55f).roundToInt()
                )

                ProfileFileCollection.writeUserAccessibility1Caption(
                    captionTextColorHex,
                    captionBgColorHex,
                    hasHaptics.value
                )

                // Update AppConfig
                AppConfig.caption_color = captionTextColorHex
                AppConfig.caption_bck_color = captionBgColorHex
                AppConfig.haptics = hasHaptics.value

                // Navigate to UserHashCachingActivity
                val intent = Intent(this, UserHashCachingActivity::class.java)
                intent.putExtra(Constants.EXTRA_HCACHING_OPTION, 1)
                startActivity(intent)
                finish()
            }
        }

        override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
            when (keyCode) {
                KeyEvent.KEYCODE_VOLUME_DOWN -> {
                    Log.d(TAG, "Volume button down pressed")
                    return true
                }

                KeyEvent.KEYCODE_VOLUME_UP -> {
                    Log.d(TAG, "Volume button up pressed")
                    return true
                }
            }
            return super.onKeyDown(keyCode, event)
        }

        override fun onDestroy() {
            super.onDestroy()
            updateRunnable?.let { updateHandler.removeCallbacks(it) }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun UserAccessibility1Screen(
        currentSection: Int,
        // BoundingBox states
        bboxRedValue: Float,
        bboxGreenValue: Float,
        bboxBlueValue: Float,
        textRedValue: Float,
        textGreenValue: Float,
        textBlueValue: Float,
        bgRedValue: Float,
        bgGreenValue: Float,
        bgBlueValue: Float,
        isBold: Boolean,
        showConfidence: Boolean,
        // Caption states
        captionTextRedValue: Float,
        captionTextGreenValue: Float,
        captionTextBlueValue: Float,
        captionBgRedValue: Float,
        captionBgGreenValue: Float,
        captionBgBlueValue: Float,
        hasHaptics: Boolean,
        previewBitmap: Bitmap?,
        // Handlers
        onBboxRedChange: (Float) -> Unit,
        onBboxGreenChange: (Float) -> Unit,
        onBboxBlueChange: (Float) -> Unit,
        onTextRedChange: (Float) -> Unit,
        onTextGreenChange: (Float) -> Unit,
        onTextBlueChange: (Float) -> Unit,
        onBgRedChange: (Float) -> Unit,
        onBgGreenChange: (Float) -> Unit,
        onBgBlueChange: (Float) -> Unit,
        onBoldChange: (Boolean) -> Unit,
        onShowConfidenceChange: (Boolean) -> Unit,
        onCaptionTextRedChange: (Float) -> Unit,
        onCaptionTextGreenChange: (Float) -> Unit,
        onCaptionTextBlueChange: (Float) -> Unit,
        onCaptionBgRedChange: (Float) -> Unit,
        onCaptionBgGreenChange: (Float) -> Unit,
        onCaptionBgBlueChange: (Float) -> Unit,
        onHapticsChange: (Boolean) -> Unit,
        onInfoClick: () -> Unit,
        onBackClick: () -> Unit,
        onNextClick: () -> Unit
    ) {
        BoxWithConstraints(
            modifier = Modifier.fillMaxSize()
        ) {
            val screenHeight = maxHeight

            // Background image
            Image(
                painter = painterResource(R.drawable.welcome_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )

            // Animated content for sections
            AnimatedContent(
                targetState = currentSection,
                transitionSpec = {
                    slideInHorizontally(
                        initialOffsetX = { if (targetState > initialState) it else -it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { if (targetState > initialState) -it else it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    )
                },
                label = "section_transition"
            ) { section ->
                when (section) {
                    1 -> BoundingBoxSection(
                        screenHeight = screenHeight,
                        bboxRedValue = bboxRedValue,
                        bboxGreenValue = bboxGreenValue,
                        bboxBlueValue = bboxBlueValue,
                        textRedValue = textRedValue,
                        textGreenValue = textGreenValue,
                        textBlueValue = textBlueValue,
                        bgRedValue = bgRedValue,
                        bgGreenValue = bgGreenValue,
                        bgBlueValue = bgBlueValue,
                        isBold = isBold,
                        showConfidence = showConfidence,
                        previewBitmap = previewBitmap,
                        onBboxRedChange = onBboxRedChange,
                        onBboxGreenChange = onBboxGreenChange,
                        onBboxBlueChange = onBboxBlueChange,
                        onTextRedChange = onTextRedChange,
                        onTextGreenChange = onTextGreenChange,
                        onTextBlueChange = onTextBlueChange,
                        onBgRedChange = onBgRedChange,
                        onBgGreenChange = onBgGreenChange,
                        onBgBlueChange = onBgBlueChange,
                        onBoldChange = onBoldChange,
                        onShowConfidenceChange = onShowConfidenceChange,
                        onInfoClick = onInfoClick
                    )

                    2 -> CaptionSection(
                        screenHeight = screenHeight,
                        captionTextRedValue = captionTextRedValue,
                        captionTextGreenValue = captionTextGreenValue,
                        captionTextBlueValue = captionTextBlueValue,
                        captionBgRedValue = captionBgRedValue,
                        captionBgGreenValue = captionBgGreenValue,
                        captionBgBlueValue = captionBgBlueValue,
                        hasHaptics = hasHaptics,
                        onCaptionTextRedChange = onCaptionTextRedChange,
                        onCaptionTextGreenChange = onCaptionTextGreenChange,
                        onCaptionTextBlueChange = onCaptionTextBlueChange,
                        onCaptionBgRedChange = onCaptionBgRedChange,
                        onCaptionBgGreenChange = onCaptionBgGreenChange,
                        onCaptionBgBlueChange = onCaptionBgBlueChange,
                        onHapticsChange = onHapticsChange,
                        onInfoClick = onInfoClick
                    )
                }
            }

            // Navigation buttons (fixed position)
            val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace),
                horizontalArrangement = Arrangement.spacedBy(30.dp)
            ) {
                BackArrowLargeFab(onClick = onBackClick)
                NextArrowLargeFab(onClick = onNextClick)
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun BoundingBoxSection(
        screenHeight: androidx.compose.ui.unit.Dp,
        bboxRedValue: Float,
        bboxGreenValue: Float,
        bboxBlueValue: Float,
        textRedValue: Float,
        textGreenValue: Float,
        textBlueValue: Float,
        bgRedValue: Float,
        bgGreenValue: Float,
        bgBlueValue: Float,
        isBold: Boolean,
        showConfidence: Boolean,
        previewBitmap: Bitmap?,
        onBboxRedChange: (Float) -> Unit,
        onBboxGreenChange: (Float) -> Unit,
        onBboxBlueChange: (Float) -> Unit,
        onTextRedChange: (Float) -> Unit,
        onTextGreenChange: (Float) -> Unit,
        onTextBlueChange: (Float) -> Unit,
        onBgRedChange: (Float) -> Unit,
        onBgGreenChange: (Float) -> Unit,
        onBgBlueChange: (Float) -> Unit,
        onBoldChange: (Boolean) -> Unit,
        onShowConfidenceChange: (Boolean) -> Unit,
        onInfoClick: () -> Unit
    ) {
        var selectedCategory by remember { mutableIntStateOf(0) }

        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.05f))

            // Preview Image
            previewBitmap?.let { bitmap ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.9f)
                        .height(screenHeight * 0.3f)
                        .background(Color.Black, RoundedCornerShape(16.dp))
                        .border(2.dp, Color.Black, RoundedCornerShape(16.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Detection Preview",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(16.dp)),
                        contentScale = ContentScale.Fit
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Segmented Button for categories
            SingleChoiceSegmentedButtonRow(
                modifier = Modifier
                    .fillMaxWidth(0.9f)
                    .shadow(3.dp, RoundedCornerShape(100.dp))
            ) {
                SegmentedButton(
                    selected = selectedCategory == 0,
                    onClick = { selectedCategory = 0 },
                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3),
                    colors = SegmentedButtonDefaults.colors(
                        activeContainerColor = colorResource(R.color.std_purple),
                        activeContentColor = Color.White,
                        inactiveContainerColor = Color.White,
                        inactiveContentColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = "Box",
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                }

                SegmentedButton(
                    selected = selectedCategory == 1,
                    onClick = { selectedCategory = 1 },
                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3),
                    colors = SegmentedButtonDefaults.colors(
                        activeContainerColor = colorResource(R.color.std_purple),
                        activeContentColor = Color.White,
                        inactiveContainerColor = Color.White,
                        inactiveContentColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = "Text",
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                }

                SegmentedButton(
                    selected = selectedCategory == 2,
                    onClick = { selectedCategory = 2 },
                    shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3),
                    colors = SegmentedButtonDefaults.colors(
                        activeContainerColor = colorResource(R.color.std_purple),
                        activeContentColor = Color.White,
                        inactiveContainerColor = Color.White,
                        inactiveContentColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Background" else "Fundal",
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Animated subsection content
            AnimatedContent(
                targetState = selectedCategory,
                transitionSpec = {
                    slideInHorizontally(
                        initialOffsetX = { if (targetState > initialState) it else -it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { if (targetState > initialState) -it else it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    )
                },
                label = "category_transition"
            ) { category ->
                when (category) {
                    0 -> ColorSliders(
                        redValue = bboxRedValue,
                        greenValue = bboxGreenValue,
                        blueValue = bboxBlueValue,
                        onRedChange = onBboxRedChange,
                        onGreenChange = onBboxGreenChange,
                        onBlueChange = onBboxBlueChange
                    )

                    1 -> ColorSliders(
                        redValue = textRedValue,
                        greenValue = textGreenValue,
                        blueValue = textBlueValue,
                        onRedChange = onTextRedChange,
                        onGreenChange = onTextGreenChange,
                        onBlueChange = onTextBlueChange
                    )

                    2 -> ColorSliders(
                        redValue = bgRedValue,
                        greenValue = bgGreenValue,
                        blueValue = bgBlueValue,
                        onRedChange = onBgRedChange,
                        onGreenChange = onBgGreenChange,
                        onBlueChange = onBgBlueChange
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Bottom controls (Bold, Show Confidence, Info)
            Row(
                modifier = Modifier.fillMaxWidth(0.9f),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Bold switch
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Bold" else "Îngroșat",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Switch(
                        checked = isBold,
                        onCheckedChange = onBoldChange,
                        colors = SwitchDefaults.colors(
                            checkedTrackColor = colorResource(R.color.std_purple),
                            checkedThumbColor = Color.White
                        )
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))

                // Show Confidence switch
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Show Confidence" else "Arată Conf.",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Switch(
                        checked = showConfidence,
                        onCheckedChange = onShowConfidenceChange,
                        colors = SwitchDefaults.colors(
                            checkedTrackColor = colorResource(R.color.std_purple),
                            checkedThumbColor = Color.White
                        )
                    )
                }

                // Info button
                IconButton(
                    onClick = onInfoClick,
                    modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Info,
                        contentDescription = "Info",
                        tint = colorResource(R.color.std_purple),
                        modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                    )
                }
            }
        }
    }

    @Composable
    fun CaptionSection(
        screenHeight: androidx.compose.ui.unit.Dp,
        captionTextRedValue: Float,
        captionTextGreenValue: Float,
        captionTextBlueValue: Float,
        captionBgRedValue: Float,
        captionBgGreenValue: Float,
        captionBgBlueValue: Float,
        hasHaptics: Boolean,
        onCaptionTextRedChange: (Float) -> Unit,
        onCaptionTextGreenChange: (Float) -> Unit,
        onCaptionTextBlueChange: (Float) -> Unit,
        onCaptionBgRedChange: (Float) -> Unit,
        onCaptionBgGreenChange: (Float) -> Unit,
        onCaptionBgBlueChange: (Float) -> Unit,
        onHapticsChange: (Boolean) -> Unit,
        onInfoClick: () -> Unit
    ) {
        var selectedCategory by remember { mutableIntStateOf(0) }

        // Calculate colors
        val textColor = Color(
            red = (captionTextRedValue * 2.55f).roundToInt(),
            green = (captionTextGreenValue * 2.55f).roundToInt(),
            blue = (captionTextBlueValue * 2.55f).roundToInt()
        )

        val bgColor = Color(
            red = (captionBgRedValue * 2.55f).roundToInt(),
            green = (captionBgGreenValue * 2.55f).roundToInt(),
            blue = (captionBgBlueValue * 2.55f).roundToInt()
        )

        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.05f))

            // Preview Box
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.9f)
                    .height(screenHeight * 0.3f)
                    .background(bgColor, RoundedCornerShape(16.dp))
                    .border(2.dp, Color.Black, RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = stringResource(R.string.std_caption),
                    fontSize = 24.sp,
                    color = textColor,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(16.dp),
                    fontFamily = robotoSemibold,
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Segmented Button for Text/Background
            SingleChoiceSegmentedButtonRow(
                modifier = Modifier
                    .fillMaxWidth(0.9f)
                    .shadow(3.dp, RoundedCornerShape(100.dp))
            ) {
                SegmentedButton(
                    selected = selectedCategory == 0,
                    onClick = { selectedCategory = 0 },
                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                    colors = SegmentedButtonDefaults.colors(
                        activeContainerColor = colorResource(R.color.std_purple),
                        activeContentColor = Color.White,
                        inactiveContainerColor = Color.White,
                        inactiveContentColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Text" else "Text",
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                }

                SegmentedButton(
                    selected = selectedCategory == 1,
                    onClick = { selectedCategory = 1 },
                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                    colors = SegmentedButtonDefaults.colors(
                        activeContainerColor = colorResource(R.color.std_purple),
                        activeContentColor = Color.White,
                        inactiveContainerColor = Color.White,
                        inactiveContentColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Background" else "Fundal",
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Animated subsection content
            AnimatedContent(
                targetState = selectedCategory,
                transitionSpec = {
                    slideInHorizontally(
                        initialOffsetX = { if (targetState > initialState) it else -it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { if (targetState > initialState) -it else it },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    )
                },
                label = "caption_category_transition"
            ) { category ->
                when (category) {
                    0 -> ColorSliders(
                        redValue = captionTextRedValue,
                        greenValue = captionTextGreenValue,
                        blueValue = captionTextBlueValue,
                        onRedChange = onCaptionTextRedChange,
                        onGreenChange = onCaptionTextGreenChange,
                        onBlueChange = onCaptionTextBlueChange
                    )

                    1 -> ColorSliders(
                        redValue = captionBgRedValue,
                        greenValue = captionBgGreenValue,
                        blueValue = captionBgBlueValue,
                        onRedChange = onCaptionBgRedChange,
                        onGreenChange = onCaptionBgGreenChange,
                        onBlueChange = onCaptionBgBlueChange
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Bottom controls (Haptics, Info)
            Row(
                modifier = Modifier.fillMaxWidth(0.9f),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Haptics switch
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en") "Haptics" else "Feedback vibrații",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Switch(
                        checked = hasHaptics,
                        onCheckedChange = onHapticsChange,
                        colors = SwitchDefaults.colors(
                            checkedTrackColor = colorResource(R.color.std_purple),
                            checkedThumbColor = Color.White
                        )
                    )
                }

                // Info button
                IconButton(
                    onClick = onInfoClick,
                    modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Info,
                        contentDescription = "Info",
                        tint = colorResource(R.color.std_purple),
                        modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                    )
                }
            }
        }
    }

    @Composable
    fun ColorSliders(
        redValue: Float,
        greenValue: Float,
        blueValue: Float,
        onRedChange: (Float) -> Unit,
        onGreenChange: (Float) -> Unit,
        onBlueChange: (Float) -> Unit
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Red Slider
            Row(
                modifier = Modifier.fillMaxWidth(0.9f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                CustomSlider(
                    value = redValue,
                    onValueChange = onRedChange,
                    valueRange = 0f..100f,
                    steps = 0,
                    thumbStyle = ThumbStyle.BAR,
                    thumbColor = Color.Red,
                    thumbWidth = 8.dp,
                    thumbHeight = 55.dp,
                    trackHeight = 25.dp,
                    trackShadow = 3.dp,
                    activeTrackColor = Color.Red,
                    inactiveTrackColor = Color.White,
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp)
                )

                Text(
                    text = "${redValue.roundToInt()} %",
                    fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                    color = colorResource(R.color.std_purple),
                    fontFamily = robotoSemibold,
                    textAlign = TextAlign.End
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Green Slider
            Row(
                modifier = Modifier.fillMaxWidth(0.9f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                CustomSlider(
                    value = greenValue,
                    onValueChange = onGreenChange,
                    valueRange = 0f..100f,
                    steps = 0,
                    thumbStyle = ThumbStyle.BAR,
                    thumbColor = colorResource(R.color.slider_green),
                    thumbWidth = 8.dp,
                    thumbHeight = 55.dp,
                    trackHeight = 25.dp,
                    trackShadow = 3.dp,
                    activeTrackColor = colorResource(R.color.slider_green),
                    inactiveTrackColor = Color.White,
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp)
                )

                Text(
                    text = "${greenValue.roundToInt()} %",
                    fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                    color = colorResource(R.color.std_purple),
                    fontFamily = robotoSemibold,
                    textAlign = TextAlign.End
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Blue Slider
            Row(
                modifier = Modifier.fillMaxWidth(0.9f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                CustomSlider(
                    value = blueValue,
                    onValueChange = onBlueChange,
                    valueRange = 0f..100f,
                    steps = 0,
                    thumbStyle = ThumbStyle.BAR,
                    thumbColor = Color.Blue,
                    thumbWidth = 8.dp,
                    thumbHeight = 55.dp,
                    trackHeight = 25.dp,
                    trackShadow = 3.dp,
                    activeTrackColor = Color.Blue,
                    inactiveTrackColor = Color.White,
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp)
                )

                Text(
                    text = "${blueValue.roundToInt()} %",
                    fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                    color = colorResource(R.color.std_purple),
                    fontFamily = robotoSemibold,
                    textAlign = TextAlign.End
                )
            }
        }
    }

    @SuppressLint("UseKtx")
    @Preview(
        name = "UserAccessibility1 Activity - BoundingBox",
        showBackground = true,
        widthDp = 412,
        heightDp = 917
    )
    @Composable
    fun UserAccessibility1BoundingBoxPreview() {
        val imageWidth = 1588f
        val imageHeight = 958f
        val mutableBitmap = Bitmap.createBitmap(
            imageWidth.toInt(),
            imageHeight.toInt(),
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(mutableBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)

        UserAccessibility1Screen(
            currentSection = 1,
            bboxRedValue = 100f,
            bboxGreenValue = 9f,
            bboxBlueValue = 50f,
            textRedValue = 100f,
            textGreenValue = 100f,
            textBlueValue = 100f,
            bgRedValue = 0f,
            bgGreenValue = 0f,
            bgBlueValue = 100f,
            isBold = true,
            showConfidence = true,
            captionTextRedValue = 100f,
            captionTextGreenValue = 0f,
            captionTextBlueValue = 0f,
            captionBgRedValue = 0f,
            captionBgGreenValue = 0f,
            captionBgBlueValue = 100f,
            hasHaptics = true,
            previewBitmap = mutableBitmap,
            onBboxRedChange = {},
            onBboxGreenChange = {},
            onBboxBlueChange = {},
            onTextRedChange = {},
            onTextGreenChange = {},
            onTextBlueChange = {},
            onBgRedChange = {},
            onBgGreenChange = {},
            onBgBlueChange = {},
            onBoldChange = {},
            onShowConfidenceChange = {},
            onCaptionTextRedChange = {},
            onCaptionTextGreenChange = {},
            onCaptionTextBlueChange = {},
            onCaptionBgRedChange = {},
            onCaptionBgGreenChange = {},
            onCaptionBgBlueChange = {},
            onHapticsChange = {},
            onInfoClick = {},
            onBackClick = {},
            onNextClick = {}
        )
    }

    @Preview(
        name = "UserAccessibility1 Activity - Caption",
        showBackground = true,
        widthDp = 412,
        heightDp = 917
    )
    @Composable
    fun UserAccessibility1CaptionPreview() {

        UserAccessibility1Screen(
            currentSection = 2,
            bboxRedValue = 0f,
            bboxGreenValue = 100f,
            bboxBlueValue = 0f,
            textRedValue = 100f,
            textGreenValue = 100f,
            textBlueValue = 100f,
            bgRedValue = 0f,
            bgGreenValue = 0f,
            bgBlueValue = 0f,
            isBold = true,
            showConfidence = true,
            captionTextRedValue = 90f,
            captionTextGreenValue = 90f,
            captionTextBlueValue = 90f,
            captionBgRedValue = 103f / 2.55f,
            captionBgGreenValue = 80f / 2.55f,
            captionBgBlueValue = 164f / 2.55f,
            hasHaptics = true,
            previewBitmap = null,
            onBboxRedChange = {},
            onBboxGreenChange = {},
            onBboxBlueChange = {},
            onTextRedChange = {},
            onTextGreenChange = {},
            onTextBlueChange = {},
            onBgRedChange = {},
            onBgGreenChange = {},
            onBgBlueChange = {},
            onBoldChange = {},
            onShowConfidenceChange = {},
            onCaptionTextRedChange = {},
            onCaptionTextGreenChange = {},
            onCaptionTextBlueChange = {},
            onCaptionBgRedChange = {},
            onCaptionBgGreenChange = {},
            onCaptionBgBlueChange = {},
            onHapticsChange = {},
            onInfoClick = {},
            onBackClick = {},
            onNextClick = {}
        )
    }