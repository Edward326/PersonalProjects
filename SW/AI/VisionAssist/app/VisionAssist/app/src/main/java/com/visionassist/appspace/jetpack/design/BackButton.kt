@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBackIos
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.visionassist.appspace.utils.Constants

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun BackArrowLargeFab(
    onClick: () -> Unit = {},
) {
    ExtendedFloatingActionButton(
        onClick = onClick,
        containerColor = Color(0xFFEADDFF),        // purple background
        contentColor  = Color(0xFF6750A4),               // icon color
        shape = RoundedCornerShape(
            topStart = 16.dp,
            topEnd = 16.dp,
            bottomEnd = 16.dp,
            bottomStart = 16.dp
        ),
        modifier = Modifier.width(Constants.NAV_BUTTONS_WIDTH.dp).height(Constants.NAV_BUTTONS_HEIGHT.dp)
    ) {
        Icon(
            imageVector = Icons.AutoMirrored.Filled.ArrowBackIos,
            contentDescription = "Back",
            modifier = Modifier.size(Constants.STD_SLIDER_INFO_SIZE.dp)
        )
    }
}

@Preview(showBackground = true, name = "Back Button", widthDp = 412, heightDp = 917)
@Composable
fun BackArrowLargeFabPreview() {
    // Center the FAB on screen for preview purposes
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        BackArrowLargeFab()
    }
}