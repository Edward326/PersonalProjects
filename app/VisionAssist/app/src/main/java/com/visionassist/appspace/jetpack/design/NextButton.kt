@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForwardIos
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun NextArrowLargeFab(
    onClick: () -> Unit = {}
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
        )
    ) {
        Icon(
            imageVector = Icons.AutoMirrored.Filled.ArrowForwardIos,
            contentDescription = "Next",
            modifier = Modifier.size(15.dp)
        )
    }
}

@Preview(showBackground = true, name = "Next Button", widthDp = 500, heightDp = 500)
@Composable
fun NextArrowLargeFabPreview() {
    // Center the FAB on screen for preview purposes
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        NextArrowLargeFab {
            // This block runs when the FAB is clicked in a real app
            println("Next FAB clicked")
        }
    }
}