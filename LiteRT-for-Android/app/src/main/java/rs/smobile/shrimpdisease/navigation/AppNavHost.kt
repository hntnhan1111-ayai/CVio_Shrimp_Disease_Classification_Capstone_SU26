package rs.smobile.shrimpdisease.navigation

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import rs.smobile.shrimpdisease.MainViewModel
import rs.smobile.shrimpdisease.RuntimeDelegate
import rs.smobile.shrimpdisease.auth.AuthResult
import rs.smobile.shrimpdisease.auth.AuthRole
import rs.smobile.shrimpdisease.ui.admin.AdminDataControlScreen
import rs.smobile.shrimpdisease.ui.admin.AdminDashboardScreen
import rs.smobile.shrimpdisease.ui.admin.AdminUsersScreen
import rs.smobile.shrimpdisease.ui.auth.AuthScreen
import rs.smobile.shrimpdisease.ui.components.AppScaffold
import rs.smobile.shrimpdisease.ui.components.ShrimpNavigationItem
import rs.smobile.shrimpdisease.ui.history.HistoryScreen
import rs.smobile.shrimpdisease.ui.home.HomeScreen
import rs.smobile.shrimpdisease.ui.inference.InferenceScreen
import rs.smobile.shrimpdisease.ui.onboarding.WelcomeScreen
import rs.smobile.shrimpdisease.ui.profile.ProfileScreen
import rs.smobile.shrimpdisease.ui.settings.SettingsScreen
import rs.smobile.shrimpdisease.utils.BitmapUtils

@Composable
fun AppNavHost(
    modifier: Modifier = Modifier,
    viewModel: MainViewModel = hiltViewModel(),
) {
    val navController = rememberNavController()
    val context = LocalContext.current

    val inputState by viewModel.inputState.collectAsStateWithLifecycle()
    val classificationState by viewModel.classificationState.collectAsStateWithLifecycle()
    val settingsState by viewModel.settingsState.collectAsStateWithLifecycle()
    val modelInfo by viewModel.modelInfo.collectAsStateWithLifecycle()
    val labels by viewModel.labels.collectAsStateWithLifecycle()
    val availableModels by viewModel.availableModels.collectAsStateWithLifecycle()
    val logs by viewModel.logs.collectAsStateWithLifecycle()
    val benchmarkMetrics by viewModel.benchmarkMetrics.collectAsStateWithLifecycle()
    val historyUiState by viewModel.historyUiState.collectAsStateWithLifecycle()
    val farmerProfileUiState by viewModel.farmerProfileUiState.collectAsStateWithLifecycle()
    val selectedGroundTruthLabel by viewModel.selectedGroundTruthLabel.collectAsStateWithLifecycle()
    val authSession by viewModel.authSession.collectAsStateWithLifecycle()
    val adminDashboardUiState by viewModel.adminDashboardUiState.collectAsStateWithLifecycle()

    var selectedAuthRole by remember { mutableStateOf(AuthRole.Farmer) }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        runCatching { BitmapUtils.decodeBitmapFromUri(context, uri) }
            .onSuccess { bitmap ->
                viewModel.setGalleryImage(uri, bitmap)
                navController.openInference()
            }
            .onFailure { error ->
                viewModel.showError(error.message ?: "Could not decode selected image.")
            }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) {
            viewModel.startCameraInput()
            navController.openInference()
        } else {
            viewModel.showError("Camera permission is required to take a snapshot.")
        }
    }

    val exportCsvLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val exported = viewModel.exportLogs(context, uri)
        Toast.makeText(
            context,
            if (exported) "Logs exported" else "Export failed",
            Toast.LENGTH_SHORT,
        ).show()
    }

    val exportAdminMetadataLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val exported = viewModel.exportAdminMetadata(context, uri)
        Toast.makeText(
            context,
            if (exported) "Admin metadata exported" else "Metadata export failed",
            Toast.LENGTH_SHORT,
        ).show()
    }

    val openCameraInput = {
        if (hasCameraPermission) {
            viewModel.startCameraInput()
            navController.openInference()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route
        ?: startDestinationFor(authSession.user?.role)
    val title = if (currentRoute == Screen.Inference.route && classificationState.result != null) {
        "Scan Result"
    } else {
        screenTitleForRoute(currentRoute)
    }

    val currentTopLevelScreens = if (authSession.user?.role == AuthRole.Admin) {
        adminTopLevelScreens
    } else {
        topLevelScreens
    }
    val navigationItems = currentTopLevelScreens.map { screen ->
        ShrimpNavigationItem(route = screen.route, label = screen.title)
    }
    val showChrome = currentRoute !in authRoutes
    val isFocusedCapture = currentRoute == Screen.Inference.route &&
        classificationState.result == null &&
        !classificationState.isLoading
    val topLevelRoutes = currentTopLevelScreens.map { it.route }
    val topLevelAnchor = if (authSession.user?.role == AuthRole.Admin) {
        Screen.AdminDashboard.route
    } else {
        Screen.Home.route
    }

    AppScaffold(
        modifier = modifier,
        title = title,
        currentRoute = currentRoute,
        navigationItems = navigationItems,
        showTopBar = showChrome && currentRoute !in setOf(
            Screen.Home.route,
            Screen.History.route,
            Screen.Profile.route,
            Screen.AdminDashboard.route,
            Screen.AdminUsers.route,
            Screen.AdminData.route,
        ),
        showBottomBar = showChrome && currentRoute in topLevelRoutes && !isFocusedCapture,
        onNavigate = { item ->
            currentTopLevelScreens.firstOrNull { it.route == item.route }?.let { screen ->
                navController.navigateTopLevel(screen, topLevelAnchor)
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = startDestinationFor(authSession.user?.role),
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.Welcome.route) {
                WelcomeScreen(
                    onGetStarted = {
                        selectedAuthRole = AuthRole.Farmer
                        navController.navigate(Screen.Auth.route)
                    },
                    onLoginAsFarmer = {
                        selectedAuthRole = AuthRole.Farmer
                        navController.navigate(Screen.Auth.route)
                    },
                    onLoginAsAdmin = {
                        selectedAuthRole = AuthRole.Admin
                        navController.navigate(Screen.Auth.route)
                    },
                )
            }

            composable(Screen.Auth.route) {
                AuthScreen(
                    selectedRole = selectedAuthRole,
                    onRoleSelected = { role -> selectedAuthRole = role },
                    onLogin = { role, account, password ->
                        when (val result = viewModel.login(role, account, password)) {
                            is AuthResult.Success -> {
                                navController.navigateAuthenticated(result.user.role)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                            }
                        }
                    },
                    onRegister = { role, account, password ->
                        when (val result = viewModel.register(role, account, password)) {
                            is AuthResult.Success -> {
                                Toast.makeText(
                                    context,
                                    "Account created",
                                    Toast.LENGTH_SHORT,
                                ).show()
                                navController.navigateAuthenticated(result.user.role)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                            }
                        }
                    },
                )
            }

            composable(Screen.Home.route) {
                HomeScreen(
                    modelInfo = modelInfo,
                    benchmarkMetrics = benchmarkMetrics,
                    logs = logs,
                    onSelectImage = { imagePicker.launch("image/*") },
                    onOpenCamera = openCameraInput,
                    onChooseModel = { navController.navigateTopLevel(Screen.Settings, Screen.Home.route) },
                )
            }

            composable(Screen.Inference.route) {
                InferenceScreen(
                    inputState = inputState,
                    classificationState = classificationState,
                    labels = labels,
                    selectedGroundTruthLabel = selectedGroundTruthLabel,
                    modelInfo = modelInfo,
                    benchmarkMetrics = benchmarkMetrics,
                    runtimeDelegateName = settingsState.runtimeDelegate.displayName,
                    debugInfoEnabled = settingsState.showDebugInfo,
                    cameraFps = settingsState.cameraFps,
                    cameraPermissionGranted = hasCameraPermission,
                    onRequestCameraPermission = {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    },
                    onOpenCamera = openCameraInput,
                    onSnapshot = { bitmap ->
                        viewModel.setCameraSnapshot(bitmap)
                    },
                    onFrameObserved = viewModel::recordCameraFrame,
                    onCameraError = viewModel::showError,
                    onGroundTruthSelected = viewModel::setGroundTruthLabel,
                    onSelectAnotherImage = { imagePicker.launch("image/*") },
                    onRunInference = viewModel::runInference,
                    onSaveResult = {
                        val hasResult = classificationState.result != null
                        val saved = viewModel.saveCurrentResult()
                        Toast.makeText(
                            context,
                            when {
                                saved -> "Result saved to history"
                                hasResult -> "Result already saved"
                                else -> "Run inference before saving"
                            },
                            Toast.LENGTH_SHORT,
                        ).show()
                    },
                    onHome = { navController.navigateHomeFromInference() },
                    onHistory = { navController.navigateTopLevel(Screen.History, Screen.Home.route) },
                )
            }

            composable(Screen.History.route) {
                HistoryScreen(
                    historyUiState = historyUiState,
                    onFilterSelected = viewModel::setHistoryFilter,
                    onExportCsv = { exportCsvLauncher.launch("shrimp_prediction_history.csv") },
                    onClearLogs = viewModel::clearLogs,
                )
            }

            composable(Screen.Profile.route) {
                ProfileScreen(
                    profileUiState = farmerProfileUiState,
                    onDataPermissionChanged = { enabled ->
                        viewModel.setFarmerDataPermission(enabled)
                    },
                    onSaveProfile = { update ->
                        val saved = viewModel.updateFarmerProfile(update)
                        Toast.makeText(
                            context,
                            if (saved) "Profile updated" else "Profile update failed",
                            Toast.LENGTH_SHORT,
                        ).show()
                    },
                    onLogout = {
                        viewModel.logout()
                        navController.navigateLoggedOut()
                    },
                )
            }

            composable(Screen.Settings.route) {
                SettingsScreen(
                    modelInfo = modelInfo,
                    labels = labels,
                    availableModels = availableModels,
                    settingsState = settingsState,
                    benchmarkMetrics = benchmarkMetrics,
                    isLoading = classificationState.isLoading,
                    errorMessage = classificationState.errorMessage,
                    onSelectModel = viewModel::loadModel,
                    onDelegateSelected = { delegate ->
                        if (delegate == RuntimeDelegate.CPU) {
                            viewModel.setRuntimeDelegate(delegate)
                        }
                    },
                    onThresholdChange = viewModel::setConfidenceThreshold,
                    onShowDebugInfoChange = viewModel::setShowDebugInfo,
                    onResetMetrics = viewModel::clearLogs,
                    userDisplayName = authSession.user?.displayName,
                    userRole = authSession.user?.role?.name,
                    onLogout = {
                        viewModel.logout()
                        navController.navigateLoggedOut()
                    },
                )
            }

            composable(Screen.AdminDashboard.route) {
                AdminDashboardScreen(
                    uiState = adminDashboardUiState,
                    currentUserName = authSession.user?.displayName,
                    onRefresh = viewModel::refreshAdminDashboard,
                )
            }

            composable(Screen.AdminUsers.route) {
                AdminUsersScreen(
                    users = adminDashboardUiState.users,
                    currentUserName = authSession.user?.displayName,
                )
            }

            composable(Screen.AdminData.route) {
                AdminDataControlScreen(
                    dataItems = adminDashboardUiState.dataItems,
                    currentUserName = authSession.user?.displayName,
                    onMarkReviewed = { item ->
                        viewModel.markAdminDataReviewed(item.id)
                        Toast.makeText(context, "Marked as reviewed", Toast.LENGTH_SHORT).show()
                    },
                    onExcludeFromTraining = { item ->
                        viewModel.excludeAdminDataFromTraining(item.id)
                        Toast.makeText(context, "Excluded from training dataset", Toast.LENGTH_SHORT).show()
                    },
                    onExportMetadata = {
                        exportAdminMetadataLauncher.launch("aquapulse_admin_metadata.csv")
                    },
                )
            }
        }
    }
}

private fun NavHostController.navigateTopLevel(screen: Screen, anchorRoute: String) {
    navigate(screen.route) {
        popUpTo(anchorRoute) {
            saveState = true
        }
        launchSingleTop = true
        restoreState = true
    }
}

private fun NavHostController.navigateAuthenticated(role: AuthRole) {
    val destination = if (role == AuthRole.Admin) {
        Screen.AdminDashboard.route
    } else {
        Screen.Home.route
    }
    navigate(destination) {
        popUpTo(Screen.Welcome.route) {
            inclusive = true
        }
        launchSingleTop = true
    }
}

private fun NavHostController.navigateLoggedOut() {
    navigate(Screen.Welcome.route) {
        popUpTo(graph.id) {
            inclusive = true
        }
        launchSingleTop = true
    }
}

private fun NavHostController.openInference() {
    navigate(Screen.Inference.route) {
        launchSingleTop = true
    }
}

private fun NavHostController.navigateHomeFromInference() {
    navigate(Screen.Home.route) {
        popUpTo(Screen.Home.route) {
            inclusive = false
        }
        launchSingleTop = true
    }
}

private fun screenTitleForRoute(route: String): String {
    return when (route) {
        Screen.Welcome.route -> "AquaPulse"
        Screen.Auth.route -> "Login"
        Screen.Home.route -> "AquaScan AI"
        Screen.Inference.route -> "Capture"
        Screen.History.route -> Screen.History.title
        Screen.Profile.route -> Screen.Profile.title
        Screen.Settings.route -> "Settings"
        Screen.AdminDashboard.route -> Screen.AdminDashboard.title
        Screen.AdminUsers.route -> Screen.AdminUsers.title
        Screen.AdminData.route -> Screen.AdminData.title
        else -> "AquaScan AI"
    }
}

private fun startDestinationFor(role: AuthRole?): String {
    return when (role) {
        AuthRole.Farmer -> Screen.Home.route
        AuthRole.Admin -> Screen.AdminDashboard.route
        null -> Screen.Welcome.route
    }
}

private val authRoutes = setOf(
    Screen.Welcome.route,
    Screen.Auth.route,
)
