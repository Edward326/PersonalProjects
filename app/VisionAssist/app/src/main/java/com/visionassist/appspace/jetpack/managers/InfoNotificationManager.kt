package com.visionassist.appspace.jetpack.managers

import android.R
import android.app.Activity
import android.view.ViewGroup
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import com.visionassist.appspace.jetpack.design.InfoNotificationDialog

class InfoNotificationManager(
    private val activity: Activity
) {
    private var isVisibleState = mutableStateOf(false)
    private var messageState = mutableStateOf("")
    private var twoButtonsState = mutableStateOf(false)
    private var firstButtonLabelState = mutableStateOf("OK")
    private var secondButtonLabelState = mutableStateOf("Cancel")
    private var onFirstButtonClickAction: (() -> Unit)? = null
    private var onSecondButtonClickAction: (() -> Unit)? = null
    private var composeView: ComposeView? = null

    /**
     * Show the info notification with a single OK button
     * Compatible with Java Runnable (void return type)
     */
    fun showNotification(
        message: String, onOkClick: Runnable, firstButtonLabel: String
    ) {
        messageState.value = message
        twoButtonsState.value = false
        firstButtonLabelState.value = firstButtonLabel
        onFirstButtonClickAction = {
                onOkClick.run()
        }

        // Lazy initialization of ComposeView
        if (composeView == null) {
            attachDialogToActivity()
        }

        isVisibleState.value = true
    }

    /**
     * Show the info notification with two buttons
     * Compatible with Java Runnable (void return type)
     */
    fun showNotificationTwoButtons(
        message: String,
        firstButtonLabel: String,
        secondButtonLabel: String,
        onFirstButtonClick: Runnable,
        onSecondButtonClick: Runnable
    ) {
        messageState.value = message
        twoButtonsState.value = true
        firstButtonLabelState.value = firstButtonLabel
        secondButtonLabelState.value = secondButtonLabel
        onFirstButtonClickAction = {
            onFirstButtonClick.run()
        }
        onSecondButtonClickAction = {
            onSecondButtonClick.run()
        }

        // Lazy initialization of ComposeView
        if (composeView == null) {
            attachDialogToActivity()
        }

        isVisibleState.value = true
    }

    /**
     * Hide the info notification
     */
    fun hideNotification() {
        isVisibleState.value = false
    }

    /**
     * Attach the ComposeView to the activity's root view
     */
    private fun attachDialogToActivity() {
        composeView = ComposeView(activity).apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)

            setContent {
                InfoNotificationDialog(
                    isVisible = isVisibleState.value,
                    message = messageState.value,
                    twoButtons = twoButtonsState.value,
                    firstButtonLabel = firstButtonLabelState.value,
                    secondButtonLabel = secondButtonLabelState.value,
                    onFirstButtonClick = { onFirstButtonClickAction?.invoke() },
                    onSecondButtonClick = { onSecondButtonClickAction?.invoke() }
                )
            }

            // Set layout params
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // Add to activity's root view
        val rootView = activity.window.decorView.findViewById<ViewGroup>(R.id.content)
        rootView.addView(composeView)
    }
}