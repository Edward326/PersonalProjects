package com.visionassist.appspace.database;

public class DBConstants {
    // ==================== FILE NAMES ====================
    private static final String PROFILE_COPY_FILENAME = "profile_copy.json";

    // ==================== STATUS CODES ====================
    public static final int STATUS_ERROR = -1;
    public static final int STATUS_INITIALIZED = 0;
    public static final int STATUS_SYNCED = 1;
    public static final int STATUS_ERROR_NETWORK = 98;
    public static final int STATUS_ERROR_DATABASE = 97;

    // ==================== OPERATION RESULTS ====================
    public static final int SYNC_NOT_NEEDED = 0;
    public static final int SYNC_OK = 1;
    public static final int SYNC_ERROR = -1;

    // Account Operations
    public static final int ACCOUNT_CREATED = 0;
    public static final int ACCOUNT_CREATION_FAILED = -1;

    // Email Operations
    public static final int EMAIL_VALID = 0;
    public static final int EMAIL_INVALID = -1;
    public static final int EMAIL_NOT_FOUND = -1;

    // Password Operations
    public static final int PASSWORD_INCORRECT = -1;
    public static final int PASSWORD_RESET_SENT = 0;

    // Data Operations
    public static final int DATA_FETCH_ERROR = -1;
    public static final int DATA_WRITE_ERROR = -1;

    // Network
    public static final int INTERNET_CONNECTION_FAILED = 99;

    // Generic
    public static final int GENERIC_ERROR = -2;

    // ==================== SYNC CONFIGURATION ====================
    // Minimum days between syncs (in days)
    public static final int SYNC_INTERVAL_DAYS = 7;

    // ==================== FIREBASE CONFIGURATION ====================
    // Firestore Collections
    public static final String FIREBASE_USERS_COLLECTION = "users";

    // ==================== FIREBASE FIELD NAMES ====================
    public static final String FIREBASE_EMAIL_FIELD = "email";
    public static final String FIREBASE_PASSWORD_FIELD = "password_hash";
    public static final String FIREBASE_PROFILE_FIELD = "profile";
    public static final String FIREBASE_LAST_SYNC_FIELD = "last_sync_date";
    public static final String FIREBASE_HASH_CACHE_ENABLED = "hash_caching";
    public static final String FIREBASE_ENV_REPORTS_ENABLED = "env_reports";

    // ==================== LOCAL FILE NAMES ====================
    public static final String PROFILE_COPY_FILE = "profile_copy.json";
    public static final String HASH_CACHE_FILE = "hash_cache.txt";
    public static final String ENV_REPORTS_FILE = "env_reports.txt";
}