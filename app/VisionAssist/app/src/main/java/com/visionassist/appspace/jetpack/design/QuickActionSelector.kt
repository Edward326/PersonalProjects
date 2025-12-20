@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Accessibility
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PlainTooltip
import androidx.compose.material3.SplitButtonDefaults
import androidx.compose.material3.SplitButtonLayout
import androidx.compose.material3.Text
import androidx.compose.material3.TooltipAnchorPosition
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.vibrate

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun QuickActionSelector(
    selectedOption: String = "Disabled",
    availableOptions: List<String> = listOf(
        "Disabled",
        "Detection-static",
        "Detection-dynamic",
        "Caption"
    ),
    onOptionSelected: (Int) -> Unit,
    manualClickExpanded: Boolean = false,
    manualExpanded: MutableState<Boolean> = mutableStateOf(false)
) {
    var expanded by remember { mutableStateOf(false) }
    var currentOption by remember { mutableStateOf(selectedOption) }
    var buttonWidth by remember { mutableIntStateOf(0) }

    val density = LocalDensity.current

    Box {
        SplitButtonLayout(
            spacing = 2.dp,
            modifier = Modifier
                .shadow(
                    elevation = 3.dp,
                    shape = MaterialTheme.shapes.extraLargeIncreased
                )
                .onGloballyPositioned { coordinates ->
                    buttonWidth = coordinates.size.width
                },
            leadingButton = {
                SplitButtonDefaults.LeadingButton(
                    enabled = false,
                    onClick = {
                        if (!manualClickExpanded) {
                            expanded = !expanded // Toggle dropdown when trailing button is clicked
                            if (AppConfig.haptics) {
                                vibrate(haptic_model0())
                            }
                        }
                    },
                    colors = ButtonDefaults.buttonColors(
                        disabledContainerColor = Color(0xFFF7F2FA),
                        disabledContentColor = Color(0xFF6750A4)
                    )
                ) {
                    Icon(
                        Icons.Filled.Accessibility,
                        modifier = Modifier.size(SplitButtonDefaults.LeadingIconSize),
                        contentDescription = "Quick Action",
                        tint = colorResource(R.color.std_cyan)
                    )
                    Spacer(Modifier.size(ButtonDefaults.IconSpacing))
                    Text(
                        currentOption,
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        fontFamily = robotoMedium,
                        fontWeight = FontWeight.Medium,
                        fontStyle = FontStyle.Normal,
                        letterSpacing = 2.sp
                    )
                }
            },
            trailingButton = {
                val description = "Quick Action Options"
                TooltipBox(
                    positionProvider = TooltipDefaults.rememberTooltipPositionProvider(
                        TooltipAnchorPosition.Above
                    ),
                    tooltip = { PlainTooltip { Text(description) } },
                    state = rememberTooltipState(),
                ) {
                    SplitButtonDefaults.TrailingButton(
                        onClick = {
                            if (!manualClickExpanded) {
                                expanded = !expanded // Toggle dropdown when trailing button is clicked
                                if (AppConfig.haptics) {
                                    vibrate(haptic_model0())
                                }
                            }
                        },
                        modifier = Modifier.semantics {
                            stateDescription = if (expanded) "Expanded" else "Collapsed"
                            contentDescription = description
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFF7F2FA),
                            contentColor = Color(0xFF6750A0)
                        )
                    ) {
                        val rotation: Float by animateFloatAsState(
                            targetValue = if (expanded) 180f else 0f,
                            animationSpec = spring(
                                dampingRatio = Spring.DampingRatioNoBouncy,
                                stiffness = Spring.StiffnessMedium
                            ),
                            label = "Trailing Icon Rotation"
                        )
                        Icon(
                            Icons.Filled.KeyboardArrowDown,
                            modifier = Modifier
                                .size(SplitButtonDefaults.TrailingIconSize)
                                .graphicsLayer {
                                    this.rotationZ = rotation
                                },
                            contentDescription = "Dropdown Arrow"
                        )
                    }
                }
            }
        )

        val trailingButtonWidth = with(density) { 48.dp.toPx() }
        val offsetX = with(density) {
            (buttonWidth - trailingButtonWidth).toDp()
        }

        DropdownMenu(
            expanded = if (manualClickExpanded) manualExpanded.value else expanded,
            onDismissRequest = {
                if (manualClickExpanded) manualExpanded.value = false else expanded = false
            },
            modifier = Modifier.wrapContentSize(),
            offset = DpOffset(x = offsetX, y = 0.dp)
        ) {
            availableOptions.forEach { option ->
                DropdownMenuItem(
                    text = {
                        Row(
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = option,
                                color = if (option == currentOption)
                                    Color(0xFF6750A4)
                                else
                                    Color(0xFF1C1B1F),
                                fontFamily = robotoMedium,
                                fontWeight = if (option == currentOption)
                                    FontWeight.SemiBold
                                else
                                    FontWeight.Normal
                            )
                        }
                    },
                    onClick = {
                        currentOption = option
                        expanded = false
                        onOptionSelected(
                            availableOptions.indexOf(option)
                        )
                    }
                )
            }
        }
    }
}

@Preview(showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun QuickActionSelectorPreview() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        QuickActionSelector(
            onOptionSelected = {}
        )
    }
}