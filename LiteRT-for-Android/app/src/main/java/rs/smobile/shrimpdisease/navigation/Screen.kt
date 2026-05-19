package rs.smobile.shrimpdisease.navigation

sealed class Screen(
    val route: String,
    val title: String,
) {
    object Welcome : Screen("welcome", "AquaPulse")
    object Auth : Screen("auth", "Login")
    object Home : Screen("home", "Home")
    object Inference : Screen("inference", "Diagnose")
    object History : Screen("history", "History")
    object Profile : Screen("profile", "Profile")
    object Settings : Screen("settings", "Settings")
    object AdminDashboard : Screen("admin_dashboard", "Dashboard")
    object AdminUsers : Screen("admin_users", "Users")
    object AdminData : Screen("admin_data", "Data")
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
    Screen.AdminData,
    Screen.Settings,
)
