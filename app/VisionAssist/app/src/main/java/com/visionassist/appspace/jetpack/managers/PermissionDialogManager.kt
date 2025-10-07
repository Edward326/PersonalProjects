package com.visionassist.appspace.jetpack.managers

import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import com.visionassist.appspace.jetpack.design.PermissionDialog

class PermissionDialogManager(
    private val dialogBox: ComposeView,
    private val initMessage: Boolean = false,
    private val context: Context
) {
    private var isVisibleState = mutableStateOf(false)
    private var messageState = mutableStateOf("Please grant the required permission")
    private var onOkClickHandler: (() -> Unit)? = null

    fun setupDialog(onOkClick: () -> Unit) {
        this.onOkClickHandler = onOkClick

        if(initMessage) {
            dialogBox.setContent {
                PermissionDialog(
                    context = context,
                    isVisible = isVisibleState.value,
                    message = messageState.value,
                    onOkClick = {
                        onOkClickHandler?.invoke()
                    }
                )
            }
        }
        else {
            dialogBox.setContent {
                PermissionDialog(
                    context = context,
                    isVisible = isVisibleState.value,
                    onOkClick = {
                        onOkClickHandler?.invoke()
                    }
                )
            }
        }
    }

    fun showDialog(message: String) {
        if(initMessage) {
            messageState.value = message
            isVisibleState.value = true
        }
    }

    fun showDialog() {
        isVisibleState.value = true
    }

    fun hideDialog() {
        isVisibleState.value = false
    }
}