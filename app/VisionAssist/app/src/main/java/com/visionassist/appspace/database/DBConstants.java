package com.visionassist.appspace.database;

public class DBConstants {

    // ==================== SUCCESS CODES ====================
    public static final int SYNC_OK = 0;
    public static final int ACCOUNT_CREATED = 0;
    public static final int EMAIL_VALID = 0;
    public static final int PASSWORD_RESET_SENT = 0;
    public static final int DATA_SYNCED = 0;

    // ==================== ERROR CODES ====================
    // Account/Email errors
    public static final int EMAIL_NOT_FOUND = -1;
    public static final int PASSWORD_INCORRECT = -1;
    public static final int EMAIL_INVALID = -1;
    public static final int ACCOUNT_NOT_FOUND = -1;

    // Generic errors
    public static final int GENERIC_ERROR = -2;
    public static final int ACCOUNT_CREATION_FAILED = -1;
    public static final int DATA_FETCH_ERROR = -1;
    public static final int DATA_WRITE_ERROR = -1;
    public static final int SYNC_ERROR = -1;

    // Network errors
    public static final int INTERNET_CONNECTION_FAILED = 99;

    // Status errors
    public static final int STATUS_ERROR = -1;

    // ==================== SYNC CONFIGURATION ====================
    // Minimum days between syncs (in days)
    public static final int SYNC_INTERVAL_DAYS = 7;

    // Firebase Configuration
    public static final String FIREBASE_USERS_COLLECTION = "users";
    public static final String FIREBASE_PROFILES_COLLECTION = "profiles";
    public static final String FIREBASE_HASH_CACHE_STORAGE = "hash_cache";
    public static final String FIREBASE_ENV_REPORTS_STORAGE = "env_reports";

    // Local File Names
    public static final String PROFILE_COPY_FILE = "profile_copy.json";
    public static final String HASH_CACHE_FILE_NAME = "hash_cache.json";
    public static final String ENV_REPORTS_FILE_NAME = "env_reports.json";

    // Firebase Field Names
    public static final String FIREBASE_EMAIL_FIELD = "email";
    public static final String FIREBASE_PASSWORD_FIELD = "password_hash";
    public static final String FIREBASE_PROFILE_FIELD = "profile";
    public static final String FIREBASE_LAST_SYNC_FIELD = "last_sync_date";
    public static final String FIREBASE_HASH_CACHE_ENABLED = "hash_caching";
    public static final String FIREBASE_ENV_REPORTS_ENABLED = "env_reports";

    // Status Messages
    public static final String STATUS_INITIALIZED = "INITIALIZED";
    public static final String STATUS_SYNCING = "SYNCING";
    public static final String STATUS_SYNCED = "SYNCED";
    public static final String STATUS_ERROR_NETWORK = "ERROR_NETWORK";
    public static final String STATUS_ERROR_DATABASE = "ERROR_DATABASE";
    public static final String STATUS_ERROR_UNKNOWN = "ERROR_UNKNOWN";
}