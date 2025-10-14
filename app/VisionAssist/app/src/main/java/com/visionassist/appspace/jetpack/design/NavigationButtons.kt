@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun NavigationButtons(
    onBackClick: () -> Unit = {},
    onNextClick: () -> Unit = {}
) {
    Row(
        modifier = Modifier
            .padding(horizontal = 120.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        BackArrowLargeFab(onClick = onBackClick)

        NextArrowLargeFab(onClick = onNextClick)
    }
}


@Preview(showBackground = true, name = "Navigation Buttons", widthDp = 412, heightDp = 917)
@Composable
fun NavigationButtonsPreview() {
    NavigationButtons(
        onBackClick = {},
        onNextClick = {}
    )
}