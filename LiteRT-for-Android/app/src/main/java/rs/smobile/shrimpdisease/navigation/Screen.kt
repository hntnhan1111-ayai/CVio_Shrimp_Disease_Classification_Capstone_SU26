package rs.smobile.shrimpdisease.navigation

sealed class Screen(
    val route: String,
    val title: String,
) {
    object Welcome : Screen("welcome", "CVio")
    object Auth : Screen("auth", "Đăng nhập")
    object Home : Screen("home", "Trang chủ")
    object Inference : Screen("inference", "Kiểm tra")
    object History : Screen("history", "Lịch sử")
    object Profile : Screen("profile", "Hồ sơ")
    object Settings : Screen("settings", "Cài đặt")
    object AdminDashboard : Screen("admin_dashboard", "Bảng điều khiển")
    object AdminUsers : Screen("admin_users", "Người dùng")
    object AdminInference : Screen("admin_inference", "Kiểm tra")
    object AdminData : Screen("admin_data", "Dữ liệu")
    object AdminLogs : Screen("admin_logs", "Nhật ký")
}

val topLevelScreens = listOf(
    Screen.Home,
    Screen.Inference,
    Screen.History,
    Screen.Profile,
)

val adminTopLevelScreens = listOf(
    Screen.AdminDashboard,
    Screen.AdminUsers,
    Screen.AdminInference,
    Screen.AdminData,
    Screen.Settings,
)
