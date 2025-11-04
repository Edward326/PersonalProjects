@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForwardIos
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.Constants

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun NextArrowLargeFab(
    onClick: () -> Unit = {}
) {
    ExtendedFloatingActionButton(
        onClick = onClick,
        containerColor = colorResource(R.color.std_light_purple),        // purple background
        contentColor  = colorResource(R.color.std_purple),               // icon color
        shape = RoundedCornerShape(
            topStart = 16.dp,
            topEnd = 16.dp,
            bottomEnd = 16.dp,
            bottomStart = 16.dp
        ),
        modifier = Modifier.width(104.dp)
    ) {
        Icon(
            imageVector = Icons.AutoMirrored.Filled.ArrowForwardIos,
            contentDescription = "Next",
            modifier = Modifier.size(Constants.STD_FONT_SIZE.dp)
        )
    }
}

@Preview(showBackground = true, name = "Next Button", widthDp = 412, heightDp = 917)
@Composable
fun NextArrowLargeFabPreview() {
    // Center the FAB on screen for preview purposes
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        NextArrowLargeFab()
    }
}