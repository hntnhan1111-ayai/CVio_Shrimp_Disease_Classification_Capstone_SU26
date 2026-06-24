#!/usr/bin/env node
'use strict';

const fs = require('node:fs');

const DEFAULT_PROJECT_ID = 'mobile-shrimpsidease';
const USERS_COLLECTION = 'users';

function parseArgs(argv) {
  const args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      args.help = true;
      continue;
    }

    if (!arg.startsWith('--')) {
      throw new Error(`Unexpected argument: ${arg}`);
    }

    const key = arg.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }

    args[key] = value;
    index += 1;
  }

  return args;
}

function printUsage() {
  console.log(`
Create or update a CVio admin account.

Required:
  --email <email>        Admin email address, or CVIO_ADMIN_EMAIL
  --password <password>  Admin password, or CVIO_ADMIN_PASSWORD

Optional:
  --name <name>          Display name, or CVIO_ADMIN_NAME
  --project <project>    Firebase project id, or FIREBASE_PROJECT_ID
                         Defaults to ${DEFAULT_PROJECT_ID}
  --credentials <path>   Firebase service account JSON path, or
                         GOOGLE_APPLICATION_CREDENTIALS

Authentication:
  Use --credentials, set GOOGLE_APPLICATION_CREDENTIALS, or run this script in
  another trusted environment with Application Default Credentials configured.
`);
}

function requireValue(name, value) {
  const normalized = String(value || '').trim();
  if (!normalized) {
    throw new Error(`Missing ${name}`);
  }
  return normalized;
}

function isPlaceholderPath(value) {
  const normalized = value.replace(/\//g, '\\').toLowerCase();
  return normalized === 'c:\\path\\to\\service-account.json' ||
    normalized.endsWith('\\path\\to\\service-account.json');
}

function resolveCredentialsPath(value) {
  const credentialsPath = String(value || '').trim();
  if (!credentialsPath) return null;

  if (isPlaceholderPath(credentialsPath)) {
    throw new Error(
      'Replace C:\\path\\to\\service-account.json with the real Firebase service account JSON path.',
    );
  }

  if (!fs.existsSync(credentialsPath)) {
    throw new Error(`Firebase service account JSON does not exist: ${credentialsPath}`);
  }

  if (!fs.statSync(credentialsPath).isFile()) {
    throw new Error(`Firebase service account path is not a file: ${credentialsPath}`);
  }

  return credentialsPath;
}

async function getOrCreateUser(auth, email, password, displayName) {
  try {
    const existing = await auth.getUserByEmail(email);
    return {
      user: await auth.updateUser(existing.uid, {
        email,
        password,
        displayName,
        disabled: false,
        emailVerified: true,
      }),
      created: false,
    };
  } catch (error) {
    if (error.code !== 'auth/user-not-found') {
      throw error;
    }

    return {
      user: await auth.createUser({
        email,
        password,
        displayName,
        disabled: false,
        emailVerified: true,
      }),
      created: true,
    };
  }
}

async function upsertAdminDocument(db, user, email, displayName) {
  const userRef = db.collection(USERS_COLLECTION).doc(user.uid);
  const snapshot = await userRef.get();
  const currentData = snapshot.exists ? snapshot.data() || {} : {};
  const currentCreatedAt = Number(currentData.createdAt);
  const authCreatedAt = Number(Date.parse(user.metadata.creationTime));
  const now = Date.now();

  await userRef.set(
    {
      id: user.uid,
      account: email,
      role: 'Admin',
      displayName,
      createdAt: Number.isFinite(currentCreatedAt)
        ? currentCreatedAt
        : Number.isFinite(authCreatedAt)
          ? authCreatedAt
          : now,
      updatedAt: now,
    },
    { merge: true },
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }

  const email = requireValue(
    'admin email',
    args.email || process.env.CVIO_ADMIN_EMAIL,
  ).toLowerCase();
  const password = requireValue(
    'admin password',
    args.password || process.env.CVIO_ADMIN_PASSWORD,
  );
  const displayName = requireValue(
    'admin display name',
    args.name || process.env.CVIO_ADMIN_NAME || 'CVio Admin',
  );
  const projectId = (
    args.project ||
    process.env.FIREBASE_PROJECT_ID ||
    process.env.GCLOUD_PROJECT ||
    DEFAULT_PROJECT_ID
  ).trim();
  const credentialsPath = resolveCredentialsPath(
    args.credentials || process.env.GOOGLE_APPLICATION_CREDENTIALS,
  );

  if (password.length < 6) {
    throw new Error('Admin password must be at least 6 characters.');
  }

  const admin = require('firebase-admin');
  const appOptions = { projectId };
  if (credentialsPath) {
    const serviceAccount = JSON.parse(fs.readFileSync(credentialsPath, 'utf8'));
    appOptions.credential = admin.credential.cert(serviceAccount);
  }
  admin.initializeApp(appOptions);

  const { user, created } = await getOrCreateUser(
    admin.auth(),
    email,
    password,
    displayName,
  );
  await upsertAdminDocument(admin.firestore(), user, email, displayName);

  console.log(`${created ? 'Created' : 'Updated'} admin account.`);
  console.log(`Project: ${projectId}`);
  if (credentialsPath) {
    console.log(`Credentials: ${credentialsPath}`);
  }
  console.log(`Email: ${email}`);
  console.log(`UID: ${user.uid}`);
  console.log(`Firestore: ${USERS_COLLECTION}/${user.uid}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exitCode = 1;
});
