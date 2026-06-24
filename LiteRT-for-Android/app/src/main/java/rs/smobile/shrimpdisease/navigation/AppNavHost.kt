package rs.smobile.shrimpdisease.navigation

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.core.tween
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
import rs.smobile.shrimpdisease.ui.components.ShrimpIconButton
import rs.smobile.shrimpdisease.ui.components.ShrimpNavigationItem
import rs.smobile.shrimpdisease.ui.components.ShrimpNavIcon
import rs.smobile.shrimpdisease.ui.history.HistoryScreen
import rs.smobile.shrimpdisease.ui.home.HomeScreen
import rs.smobile.shrimpdisease.ui.inference.InferenceScreen
import rs.smobile.shrimpdisease.ui.onboarding.WelcomeScreen
import rs.smobile.shrimpdisease.ui.profile.ProfileScreen
import rs.smobile.shrimpdisease.ui.settings.AdminLogsScreen
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
    val adminModelConfigUiState by viewModel.adminModelConfigUiState.collectAsStateWithLifecycle()
    val adminInferenceLogsUiState by viewModel.adminInferenceLogsUiState.collectAsStateWithLifecycle()

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
                navController.openInference(authSession.user?.role)
            }
            .onFailure { error ->
                viewModel.showError(error.message ?: "Không đọc được ảnh đã chọn.")
            }
    }

    val avatarPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        viewModel.updateFarmerAvatar(uri.toString()) { updated ->
        Toast.makeText(
            context,
            if (updated) "Đã cập nhật ảnh đại diện" else "Cập nhật ảnh đại diện thất bại",
            Toast.LENGTH_SHORT,
        ).show()
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) {
            viewModel.startCameraInput()
            navController.openInference(authSession.user?.role)
        } else {
            viewModel.showError("Cần quyền máy ảnh để chụp ảnh tôm.")
        }
    }

    val exportCsvLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val exported = viewModel.exportLogs(context, uri)
        Toast.makeText(
            context,
            if (exported) "Đã xuất lịch sử" else "Xuất lịch sử thất bại",
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
            if (exported) "Đã xuất metadata quản trị" else "Xuất metadata thất bại",
            Toast.LENGTH_SHORT,
        ).show()
    }

    val openCameraInput = {
        if (hasCameraPermission) {
            viewModel.startCameraInput()
            navController.openInference(authSession.user?.role)
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route
        ?: startDestinationFor(authSession.user?.role)
    val isInferenceRoute = currentRoute == Screen.Inference.route ||
        currentRoute == Screen.AdminInference.route
    val title = if (isInferenceRoute && classificationState.result != null) {
        "Kết quả kiểm tra"
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
    val isFocusedCapture = isInferenceRoute &&
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
            Screen.AdminInference.route,
            Screen.AdminData.route,
        ),
        showBottomBar = showChrome && currentRoute in topLevelRoutes && !isFocusedCapture,
        topBarNavigationIcon = when (currentRoute) {
            Screen.Inference.route -> {
                {
                    ShrimpIconButton(
                        icon = ShrimpNavIcon.Home,
                        contentDescription = "Về trang chủ",
                        onClick = { navController.navigateHomeFromInference() },
                    )
                }
            }

            Screen.AdminInference.route -> {
                {
                    ShrimpIconButton(
                        icon = ShrimpNavIcon.Home,
                        contentDescription = "Về trang chủ",
                        onClick = { navController.navigateAdminHomeFromInference() },
                    )
                }
            }

            Screen.Settings.route -> {
                {
                    ShrimpIconButton(
                        icon = ShrimpNavIcon.Back,
                        contentDescription = "Về trang chủ",
                        onClick = {
                            if (authSession.user?.role == AuthRole.Admin) {
                                navController.navigateTopLevel(
                                    Screen.AdminDashboard,
                                    Screen.AdminDashboard.route,
                                )
                            } else {
                                navController.navigateTopLevel(Screen.Home, Screen.Home.route)
                            }
                        },
                    )
                }
            }

            Screen.AdminLogs.route -> {
                {
                    ShrimpIconButton(
                        icon = ShrimpNavIcon.Back,
                        contentDescription = "Quay lại",
                        onClick = {
                            if (!navController.popBackStack()) {
                                navController.navigateAdminHomeFromInference()
                            }
                        },
                    )
                }
            }

            else -> null
        },
        onNavigate = { item ->
            if (item.route != currentRoute) {
                when (item.route) {
                    Screen.Home.route -> navController.navigateHomeFromInference()
                    else -> currentTopLevelScreens.firstOrNull { it.route == item.route }?.let { screen ->
                        navController.navigateTopLevel(screen, topLevelAnchor)
                    }
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = startDestinationFor(authSession.user?.role),
            modifier = Modifier.padding(innerPadding),
            enterTransition = {
                pageEnterTransition(
                    forward = isForwardNavigation(
                        fromRoute = initialState.destination.route,
                        toRoute = targetState.destination.route,
                        role = authSession.user?.role,
                    ),
                )
            },
            exitTransition = {
                pageExitTransition(
                    forward = isForwardNavigation(
                        fromRoute = initialState.destination.route,
                        toRoute = targetState.destination.route,
                        role = authSession.user?.role,
                    ),
                )
            },
            popEnterTransition = {
                pageEnterTransition(
                    forward = isForwardNavigation(
                        fromRoute = initialState.destination.route,
                        toRoute = targetState.destination.route,
                        role = authSession.user?.role,
                    ),
                )
            },
            popExitTransition = {
                pageExitTransition(
                    forward = isForwardNavigation(
                        fromRoute = initialState.destination.route,
                        toRoute = targetState.destination.route,
                        role = authSession.user?.role,
                    ),
                )
            },
        ) {
            composable(Screen.Welcome.route) {
                WelcomeScreen(
                    onGetStarted = {
                        navController.navigate(Screen.Auth.route)
                    },
                )
            }

            composable(Screen.Auth.route) {
                AuthScreen(
                    onLogin = { account, password ->
                        viewModel.login(account, password) { result ->
                        when (result) {
                            is AuthResult.Success -> {
                                navController.navigateAuthenticated(result.user.role)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                            }
                        }
                        }
                    },
                    onRegister = { account, password ->
                        viewModel.register(account, password) { result ->
                        when (result) {
                            is AuthResult.Success -> {
                                Toast.makeText(
                                    context,
                                    "Đã tạo tài khoản",
                                    Toast.LENGTH_SHORT,
                                ).show()
                                navController.navigateAuthenticated(result.user.role)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                            }
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
                    onOpenProfile = {
                        navController.navigateTopLevel(Screen.Profile, Screen.Home.route)
                    },
                    onViewAllHistory = {
                        navController.navigateTopLevel(Screen.History, Screen.Home.route)
                    },
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
                        viewModel.saveCurrentResult { saved, hasResult ->
                        Toast.makeText(
                            context,
                            when {
                                saved -> "Đã lưu kết quả vào lịch sử"
                                hasResult -> "Kết quả đã được lưu"
                                else -> "Hãy kiểm tra ảnh trước khi lưu"
                            },
                            Toast.LENGTH_SHORT,
                        ).show()
                        }
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
                    onBackHome = {
                        navController.navigateTopLevel(Screen.Home, Screen.Home.route)
                    },
                    onSettings = {
                        navController.navigateTopLevel(Screen.Settings, Screen.Home.route)
                    },
                )
            }

            composable(Screen.Profile.route) {
                ProfileScreen(
                    profileUiState = farmerProfileUiState,
                    onDataPermissionChanged = { enabled ->
                        viewModel.setFarmerDataPermission(enabled)
                    },
                    onBackHome = {
                        navController.navigateTopLevel(Screen.Home, Screen.Home.route)
                    },
                    onSettings = {
                        navController.navigateTopLevel(Screen.Settings, Screen.Home.route)
                    },
                    onUpdateAvatar = {
                        avatarPicker.launch("image/*")
                    },
                    onSaveProfile = { update ->
                        viewModel.updateFarmerProfile(update) { saved ->
                        Toast.makeText(
                            context,
                            if (saved) "Đã cập nhật hồ sơ" else "Cập nhật hồ sơ thất bại",
                            Toast.LENGTH_SHORT,
                        ).show()
                        }
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
                    currentResult = classificationState.result,
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
                    adminModelConfigUiState = adminModelConfigUiState,
                    adminInferenceLogsUiState = adminInferenceLogsUiState,
                    onAdminModelConfigSave = { update ->
                        viewModel.saveAdminModelConfig(update)
                        Toast.makeText(context, "Đã lưu cấu hình mô hình", Toast.LENGTH_SHORT).show()
                    },
                    onAdminModelDeploy = { modelFile ->
                        viewModel.deployAdminModel(modelFile)
                        Toast.makeText(context, "Đã triển khai mô hình", Toast.LENGTH_SHORT).show()
                    },
                    onOpenAdminLogs = {
                        navController.navigate(Screen.AdminLogs.route) {
                            launchSingleTop = true
                        }
                    },
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
                    onOpenSettings = {
                        navController.navigateTopLevel(Screen.Settings, Screen.AdminDashboard.route)
                    },
                    onOpenUsers = {
                        navController.navigateTopLevel(Screen.AdminUsers, Screen.AdminDashboard.route)
                    },
                    onOpenData = {
                        navController.navigateTopLevel(Screen.AdminData, Screen.AdminDashboard.route)
                    },
                    onOpenInference = {
                        navController.navigateTopLevel(Screen.AdminInference, Screen.AdminDashboard.route)
                    },
                )
            }

            composable(Screen.AdminUsers.route) {
                AdminUsersScreen(
                    users = adminDashboardUiState.users,
                    currentUserName = authSession.user?.displayName,
                    onCreateUser = { input, onComplete ->
                        viewModel.createAdminUser(input) { result ->
                        when (result) {
                            is AuthResult.Success -> {
                                Toast.makeText(
                                    context,
                                    "Đã tạo người dùng",
                                    Toast.LENGTH_SHORT,
                                ).show()
                                onComplete(true)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                                onComplete(false)
                            }
                        }
                        }
                    },
                    onUpdateUser = { userId, input, onComplete ->
                        viewModel.updateAdminUser(userId, input) { result ->
                        when (result) {
                            is AuthResult.Success -> {
                                Toast.makeText(context, "Đã cập nhật người dùng", Toast.LENGTH_SHORT).show()
                                onComplete(true)
                            }

                            is AuthResult.Error -> {
                                Toast.makeText(context, result.message, Toast.LENGTH_SHORT).show()
                                onComplete(false)
                            }
                        }
                        }
                    },
                    onDeleteUser = { user, onComplete ->
                        viewModel.deleteAdminUser(user.id) { deleted ->
                        Toast.makeText(
                            context,
                            if (deleted) "Đã xóa người dùng" else "Không xóa được người dùng",
                            Toast.LENGTH_SHORT,
                        ).show()
                        onComplete(deleted)
                        }
                    },
                    onHome = {
                        navController.navigateTopLevel(Screen.AdminDashboard, Screen.AdminDashboard.route)
                    },
                    onSettings = {
                        navController.navigateTopLevel(Screen.Settings, Screen.AdminDashboard.route)
                    },
                )
            }

            composable(Screen.AdminInference.route) {
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
                        viewModel.saveCurrentResult { saved, hasResult ->
                        Toast.makeText(
                            context,
                            when {
                                saved -> "Đã lưu kết quả vào lịch sử"
                                hasResult -> "Kết quả đã được lưu"
                                else -> "Hãy kiểm tra ảnh trước khi lưu"
                            },
                            Toast.LENGTH_SHORT,
                        ).show()
                        }
                    },
                    onHome = { navController.navigateAdminHomeFromInference() },
                    onHistory = {
                        navController.navigate(Screen.AdminLogs.route) {
                            launchSingleTop = true
                        }
                    },
                    historyActionText = "Nhật ký",
                    showBackButton = true,
                    onBack = { navController.navigateAdminHomeFromInference() },
                    enableGroundTruthSelection = true,
                    showResultMetrics = true,
                )
            }

            composable(Screen.AdminData.route) {
                AdminDataControlScreen(
                    dataItems = adminDashboardUiState.dataItems,
                    currentUserName = authSession.user?.displayName,
                    onCreateData = { input, onComplete ->
                        viewModel.createAdminData(input) { created ->
                        Toast.makeText(
                            context,
                            if (created) "Đã thêm dữ liệu" else "Không thêm được dữ liệu",
                            Toast.LENGTH_SHORT,
                        ).show()
                        onComplete(created)
                        }
                    },
                    onUpdateData = { item, input, onComplete ->
                        viewModel.updateAdminData(item.id, input) { updated ->
                        Toast.makeText(
                            context,
                            if (updated) "Đã cập nhật dữ liệu" else "Không cập nhật được dữ liệu",
                            Toast.LENGTH_SHORT,
                        ).show()
                        onComplete(updated)
                        }
                    },
                    onDeleteData = { item, onComplete ->
                        viewModel.deleteAdminData(item.id) { deleted ->
                        Toast.makeText(
                            context,
                            if (deleted) "Đã xóa dữ liệu" else "Không xóa được dữ liệu",
                            Toast.LENGTH_SHORT,
                        ).show()
                        onComplete(deleted)
                        }
                    },
                    onMarkReviewed = { item ->
                        viewModel.markAdminDataReviewed(item.id)
                        Toast.makeText(context, "Đã đánh dấu duyệt", Toast.LENGTH_SHORT).show()
                    },
                    onExcludeFromTraining = { item ->
                        viewModel.excludeAdminDataFromTraining(item.id)
                        Toast.makeText(context, "Đã loại khỏi tập huấn luyện", Toast.LENGTH_SHORT).show()
                    },
                    onExportMetadata = {
                        exportAdminMetadataLauncher.launch("cvio_admin_metadata.csv")
                    },
                    onHome = {
                        navController.navigateTopLevel(Screen.AdminDashboard, Screen.AdminDashboard.route)
                    },
                    onSettings = {
                        navController.navigateTopLevel(Screen.Settings, Screen.AdminDashboard.route)
                    },
                )
            }

            composable(Screen.AdminLogs.route) {
                AdminLogsScreen(
                    inferenceLogs = adminInferenceLogsUiState,
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

private fun NavHostController.openInference(role: AuthRole?) {
    val route = if (role == AuthRole.Admin) {
        Screen.AdminInference.route
    } else {
        Screen.Inference.route
    }
    navigate(route) {
        launchSingleTop = true
    }
}

private fun NavHostController.navigateHomeFromInference() {
    if (popBackStack(Screen.Home.route, inclusive = false)) return

    val currentDestinationId = currentDestination?.id
    navigate(Screen.Home.route) {
        if (currentDestinationId != null) {
            popUpTo(currentDestinationId) {
                inclusive = true
            }
        }
        launchSingleTop = true
    }
}

private fun NavHostController.navigateAdminHomeFromInference() {
    if (popBackStack(Screen.AdminDashboard.route, inclusive = false)) return

    val currentDestinationId = currentDestination?.id
    navigate(Screen.AdminDashboard.route) {
        if (currentDestinationId != null) {
            popUpTo(currentDestinationId) {
                inclusive = true
            }
        }
        launchSingleTop = true
    }
}

private fun screenTitleForRoute(route: String): String {
    return when (route) {
        Screen.Welcome.route -> "CVio"
        Screen.Auth.route -> "Đăng nhập"
        Screen.Home.route -> "CVio AI"
        Screen.Inference.route -> "Chụp ảnh"
        Screen.History.route -> Screen.History.title
        Screen.Profile.route -> Screen.Profile.title
        Screen.Settings.route -> "Cài đặt"
        Screen.AdminDashboard.route -> Screen.AdminDashboard.title
        Screen.AdminUsers.route -> Screen.AdminUsers.title
        Screen.AdminInference.route -> Screen.AdminInference.title
        Screen.AdminData.route -> Screen.AdminData.title
        Screen.AdminLogs.route -> Screen.AdminLogs.title
        else -> "CVio AI"
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

private const val NavAnimationMillis = 260

private fun pageEnterTransition(forward: Boolean): EnterTransition {
    return fadeIn(animationSpec = tween(durationMillis = NavAnimationMillis)) +
        slideInHorizontally(animationSpec = tween(durationMillis = NavAnimationMillis)) { width ->
            if (forward) width / 5 else -width / 5
        }
}

private fun pageExitTransition(forward: Boolean): ExitTransition {
    return fadeOut(animationSpec = tween(durationMillis = NavAnimationMillis - 60)) +
        slideOutHorizontally(animationSpec = tween(durationMillis = NavAnimationMillis)) { width ->
            if (forward) -width / 6 else width / 6
        }
}

private fun isForwardNavigation(
    fromRoute: String?,
    toRoute: String?,
    role: AuthRole?,
): Boolean {
    val routeOrder = routeOrderFor(role)
    val fromIndex = routeOrder.indexOf(fromRoute)
    val toIndex = routeOrder.indexOf(toRoute)
    return if (fromIndex >= 0 && toIndex >= 0) {
        toIndex >= fromIndex
    } else {
        true
    }
}

private fun routeOrderFor(role: AuthRole?): List<String> {
    return when (role) {
        AuthRole.Admin -> listOf(
            Screen.Welcome.route,
            Screen.Auth.route,
            Screen.AdminDashboard.route,
            Screen.AdminUsers.route,
            Screen.AdminInference.route,
            Screen.AdminData.route,
            Screen.AdminLogs.route,
            Screen.Settings.route,
        )

        AuthRole.Farmer -> listOf(
            Screen.Welcome.route,
            Screen.Auth.route,
            Screen.Home.route,
            Screen.Inference.route,
            Screen.History.route,
            Screen.Profile.route,
            Screen.Settings.route,
        )

        null -> listOf(
            Screen.Welcome.route,
            Screen.Auth.route,
        )
    }
}
