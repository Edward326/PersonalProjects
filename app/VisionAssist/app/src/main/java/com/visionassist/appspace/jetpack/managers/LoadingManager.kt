package com.visionassist.appspace.jetpack.managers

import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import com.visionassist.appspace.jetpack.design.LoadingComponent

class LoadingManager(
    private val loadingBox: ComposeView,
    private val initMessage: Boolean = false,
    private val context: Context
) {
    private var isVisibleState = mutableStateOf(false)
    private var currentMessageState = mutableStateOf("Please wait")

    /**
     * Initializes the ComposeView with the LoadingComponent.
     * This needs to be called once in onCreate.
     */
    fun setupLoadingBox() {
        loadingBox.setContent {
            if (initMessage) {
                LoadingComponent(
                    isVisible = isVisibleState.value,
                    loadingText = currentMessageState.value,
                    context=context
                )
            } else {
                LoadingComponent(
                    isVisible = isVisibleState.value,
                    context=context
                )
            }
        }
        // Set the initial visibility to GONE
        //loadingBox.visibility = View.GONE
        //isVisibleState.value = false
    }

    /**
     * Displays the loading screen with the default message (mainLanguage)
     */
    fun showLoading() {
        if (!initMessage) {
            //loadingBox.visibility = View.VISIBLE
            isVisibleState.value = true
        }
    }

    /**
     * Displays the loading screen with a custom message.
     */
    fun showLoading(message: String) {
        if (initMessage) {
            currentMessageState.value = message
            //loadingBox.visibility = View.VISIBLE
            isVisibleState.value = true
        }
    }

    /**
     * Hides the loading screen with fade out animation.
     */
    fun hideLoading() {
        isVisibleState.value = false
        //loadingBox.visibility = View.GONE
    }
}