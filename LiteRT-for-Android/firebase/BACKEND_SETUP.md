# CVio Firebase Backend Setup

The Android app can sign in with Firebase Auth and sync metadata through
Firestore, but it cannot safely create the first admin account by itself. Create
the first admin with Firebase Console or a trusted Admin SDK script, then deploy
the Firestore rules in this folder.

## 1. Create The First Admin With A Script

Install the trusted Firebase helper dependencies once:

```powershell
cd LiteRT-for-Android\firebase
npm.cmd install
```

Download a Firebase service account JSON from Firebase Console:

Project settings > Service accounts > Generate new private key

Keep this JSON file private. Do not commit it to git.

Then run:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\secure\mobile-shrimpsidease-service-account.json"
$env:CVIO_ADMIN_EMAIL="admin@example.com"
$env:CVIO_ADMIN_PASSWORD="ChangeMe123!"
$env:CVIO_ADMIN_NAME="CVio Admin"
npm.cmd run create:admin
```

Or pass the JSON path directly:

```powershell
npm.cmd run create:admin -- --credentials "D:\secure\mobile-shrimpsidease-service-account.json" --email "admin@example.com" --password "ChangeMe123!" --name "CVio Admin"
```

The script creates or updates the Firebase Auth user, then writes the matching
Firestore document at `users/{AUTH_UID}` with `role = "Admin"`.

## 2. Create The First Admin Manually

1. Open Firebase Console for project `mobile-shrimpsidease`.
2. Go to Authentication > Users.
3. Add an email/password user for the admin account.
4. Copy the new Firebase Auth UID.
5. Go to Firestore Database and create this document:

Collection: `users`

Document ID: the Firebase Auth UID

Fields:

```json
{
  "id": "<AUTH_UID>",
  "account": "admin@example.com",
  "role": "Admin",
  "displayName": "CVio Admin",
  "createdAt": 1780000000000,
  "updatedAt": 1780000000000
}
```

Use the current Unix time in milliseconds for `createdAt` and `updatedAt`.

## 3. Deploy Firestore Rules

From `LiteRT-for-Android/firebase`:

```powershell
firebase deploy --only firestore:rules
```

## 4. Debug Versus Release Behavior

Debug builds can seed a local admin for offline development when a password is
provided through `CVIO_DEFAULT_ADMIN_PASSWORD` as a Gradle property or
environment variable. The default account is:

```text
admin@cvio.local
```

Release builds do not seed or accept local auth fallback. A release admin must
exist in Firebase Auth and have a matching `users/{uid}` document with
`role = "Admin"`.

## 5. Admin-Created Farmers

Creating Firebase Auth users for other people requires a trusted backend
environment, such as Cloud Functions using the Firebase Admin SDK. The Android
client cannot securely create managed Firebase Auth users while staying signed
in as the admin.

Until Cloud Functions are added, farmer self-registration is the cloud-ready
path. Debug builds can still use local managed users for development.
