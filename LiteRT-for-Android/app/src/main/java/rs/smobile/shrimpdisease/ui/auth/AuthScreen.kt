package rs.smobile.shrimpdisease.ui.auth

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.ui.components.PrimaryActionButton
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainer
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLow
import rs.smobile.shrimpdisease.ui.theme.CVioSurfaceContainerLowest

@Composable
fun AuthScreen(
    selectedRole: AuthRole = AuthRole.Farmer,
    onRoleSelected: (AuthRole) -> Unit,
    onLogin: (AuthRole, String, String) -> Unit,
    onRegister: (AuthRole, String, String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var role by remember(selectedRole) { mutableStateOf(selectedRole) }
    var account by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var showPassword by remember { mutableStateOf(false) }
    var registerMode by remember { mutableStateOf(false) }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(20.dp),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.extraLarge,
            color = CVioSurfaceContainerLowest,
            shadowElevation = 10.dp,
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(24.dp),
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .size(64.dp)
                            .clip(RoundedCornerShape(999.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "AP",
                            style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Text(
                        text = "AquaPulse",
                        style = MaterialTheme.typography.displayLarge,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = if (registerMode) {
                            "Create a farmer account to start diagnosis"
                        } else {
                            "Sign in to continue to your dashboard"
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                    )
                }

                RoleSelector(
                    selectedRole = role,
                    onRoleSelected = { selected ->
                        role = selected
                        onRoleSelected(selected)
                    },
                )

                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    OutlinedTextField(
                        value = account,
                        onValueChange = { account = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text(text = "Email or Phone Number") },
                        placeholder = { Text(text = "Enter your details") },
                        singleLine = true,
                        shape = RoundedCornerShape(24.dp),
                    )
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = "Password",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            if (!registerMode) {
                                TextButton(onClick = { }) {
                                    Text(text = "Forgot?")
                                }
                            }
                        }
                        OutlinedTextField(
                            value = password,
                            onValueChange = { password = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text(text = "Enter your password") },
                            singleLine = true,
                            shape = RoundedCornerShape(24.dp),
                            visualTransformation = if (showPassword) {
                                VisualTransformation.None
                            } else {
                                PasswordVisualTransformation()
                            },
                            trailingIcon = {
                                TextButton(onClick = { showPassword = !showPassword }) {
                                    Text(text = if (showPassword) "Hide" else "Show")
                                }
                            },
                        )
                    }
                }

                PrimaryActionButton(
                    text = if (registerMode) "Create account" else "Login",
                    onClick = {
                        if (registerMode) {
                            onRegister(role, account, password)
                        } else {
                            onLogin(role, account, password)
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                )

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(24.dp))
                        .background(CVioSurfaceContainerLow)
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = if (registerMode) {
                            "Already have an account? Login"
                        } else {
                            "New to AquaPulse? Create farmer account"
                        },
                        modifier = Modifier.clickable { registerMode = !registerMode },
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold,
                        textAlign = TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Composable
private fun RoleSelector(
    selectedRole: AuthRole,
    onRoleSelected: (AuthRole) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(999.dp))
            .background(CVioSurfaceContainer)
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        AuthRole.values().forEach { role ->
            val selected = selectedRole == role
            Surface(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(999.dp))
                    .clickable { onRoleSelected(role) },
                shape = RoundedCornerShape(999.dp),
                color = if (selected) {
                    CVioSurfaceContainerLowest
                } else {
                    Color.Transparent
                },
                shadowElevation = if (selected) 2.dp else 0.dp,
            ) {
                Text(
                    text = role.name,
                    modifier = Modifier.padding(vertical = 12.dp),
                    style = MaterialTheme.typography.labelMedium,
                    color = if (selected) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    textAlign = TextAlign.Center,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}
