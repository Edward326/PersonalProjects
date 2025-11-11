package com.visionassist.appspace.database;

public class DBConstants {
    // ==================== FILE NAMES ====================
    public static final String PROFILE_COPY_FILE = "profile_copy.json";

    // ==================== STATUS CODES ====================
    public static final int STATUS_INITIALIZED = 0;

    // ==================== OPERATION RESULTS ====================
    public static final int SYNC_NOT_NEEDED = 1000;
    public static final int SYNC_OK = 1;
    public static final int SYNC_ERROR = -1;

    // Account Operations
    public static final int ACCOUNT_CREATED = 9999;
    public static final int ACCOUNT_CREATION_FAILED = -9999;

    // Email Operations
    public static final int EMAIL_VALID = 5555;
    public static final int EMAIL_INVALID = -5555;
    public static final int EMAIL_NOT_FOUND = -8888;

    // Password Operations
    public static final int PASSWORD_INCORRECT = -7777;
    public static final int PASSWORD_RESET_SENT = 7777;

    // Data Operations
    public static final int DATA_FETCH_ERROR = -6666;
    public static final int DATA_WRITE_ERROR = -5555;

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
    public static final String FIREBASE_LAST_SYNC_FIELD = "last_sync_date";
}