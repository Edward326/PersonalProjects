package com.visionassist.appspace.jetpack.managers

import android.app.Activity
import android.view.ViewGroup
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import com.visionassist.appspace.jetpack.design.ErrorDialog

class ErrorDialogManager(
    private val activity: Activity
) {
    private var isVisibleState = mutableStateOf(false)
    private var errorCodeState = mutableStateOf(0)
    private var messageState = mutableStateOf("_undefined_")
    private var composeView: ComposeView? = null
    private var isSetup = false

    /**
     * Setup dialog with error code only (uses default message from resources)
     * Call this once during initialization
     */
    fun setupDialog(errorCode: Int) {
        errorCodeState.value = errorCode
        messageState.value = "_undefined_"
        attachDialogToActivity()
        isSetup = true
    }

    /**
     * Setup dialog with custom message and error code
     * Call this once during initialization
     */
    fun setupDialog(errorCode: Int, message: String) {
        errorCodeState.value = errorCode
        messageState.value = message
        attachDialogToActivity()
        isSetup = true
    }

    /**
     * Show the dialog (must call setupDialog first)
     */
    fun showDialog() {
        if (!isSetup) {
            throw IllegalStateException("Must call setupDialog() before showDialog()")
        }
        isVisibleState.value = true
    }

    /**
     * Hide the error dialog
     */
    fun hideDialog() {
        isVisibleState.value = false
    }

    /**
     * Attach the ComposeView to the activity's root view
     */
    private fun attachDialogToActivity() {
        if (composeView == null) {
            composeView = ComposeView(activity).apply {
                setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)

                setContent {
                    ErrorDialog(
                        context = activity,
                        isVisible = isVisibleState.value,
                        message = messageState.value,
                        errorCode = errorCodeState.value
                    )
                }

                // Add to activity's root view
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }

            // Get the activity's root view and add our ComposeView on top
            val rootView = activity.window.decorView.findViewById<ViewGroup>(android.R.id.content)
            rootView.addView(composeView)
        }
    }

    /**
     * Remove the ComposeView from the activity
     */
    private fun detachDialogFromActivity() {
        composeView?.let { view ->
            val rootView = activity.window.decorView.findViewById<ViewGroup>(android.R.id.content)
            rootView.removeView(view)
            composeView = null
        }
        isSetup = false
    }

    /**
     * Clean up resources (call in Activity's onDestroy)
     */
    fun cleanup() {
        hideDialog()
        detachDialogFromActivity()
    }
}