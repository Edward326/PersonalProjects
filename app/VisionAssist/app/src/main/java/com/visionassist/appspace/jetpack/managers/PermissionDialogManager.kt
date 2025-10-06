package com.visionassist.appspace.jetpack.managers

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import com.visionassist.appspace.jetpack.design.PermissionDialog

class PermissionDialogManager(
    private val dialogBox: ComposeView
) {
    private var isVisibleState = mutableStateOf(false)
    private var titleState = mutableStateOf("Permission Required")
    private var messageState = mutableStateOf("Please grant the required permission")
    private var onOkClickHandler: (() -> Unit)? = null

    /**
     * Initializes the ComposeView with the PermissionDialog.
     * This needs to be called once in onCreate.
     * @param onOkClick The function to be triggered when OK button is clicked
     */
    fun setupDialog(onOkClick: () -> Unit) {
        this.onOkClickHandler = onOkClick

        dialogBox.setContent {
            PermissionDialog(
                isVisible = isVisibleState.value,
                title = titleState.value,
                message = messageState.value,
                onOkClick = {
                    onOkClickHandler?.invoke()
                }
            )
        }
    }

    /**
     * Shows the permission dialog with custom title and message
     * @param title Dialog title
     * @param message Dialog message
     */
    fun showDialog(title: String, message: String) {
        titleState.value = title
        messageState.value = message
        isVisibleState.value = true
    }

    /**
     * Shows the permission dialog with default title
     * @param message Dialog message
     */
    fun showDialog(message: String) {
        messageState.value = message
        isVisibleState.value = true
    }

    /**
     * Hides the permission dialog with fade out animation
     */
    fun hideDialog() {
        isVisibleState.value = false
    }

    /**
     * Updates the onClick handler dynamically
     * @param onOkClick New click handler function
     */
    fun setOnClickHandler(onOkClick: () -> Unit) {
        this.onOkClickHandler = onOkClick
    }
}