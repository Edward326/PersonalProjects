package com.visionassist.appspace.database;

import android.content.Context;
import android.util.Log;
import android.util.Pair;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.utils.FileUtils;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;

public class DBManager {
    private static final String TAG = "DBManager";

    private Context context;
    private FirebaseAuth firebaseAuth;
    private FirebaseFirestore firebaseDb;
    private FirebaseStorage firebaseStorage;
    private String status = DBConstants.STATUS_INITIALIZED;
    private String currentAccountHash = null;

    public DBManager(Context context) {
        this.context = context;
        this.firebaseAuth = FirebaseAuth.getInstance();
        this.firebaseDb = FirebaseFirestore.getInstance();
        this.firebaseStorage = FirebaseStorage.getInstance();
        Log.d(TAG, "DBManager initialized");
    }

    /** @noinspection BooleanMethodIsAlwaysInverted*/
    private boolean hasInternetConnection() {
        return NetworkUtils.isNetworkConnected(context);
    }

    public Pair<Integer, String> verifyAccount(String email, String password) {
        Log.d(TAG, "Verifying account for email: " + email);

        // Check internet
        if (!hasInternetConnection()) {
            status = DBConstants.STATUS_ERROR_NETWORK;
            Log.w(TAG, "No internet connection");
            return new Pair<>(DBConstants.INTERNET_CONNECTION_FAILED, null);
        }

        try {
            // Check if email exists in database
            if (!emailExistsInDatabase(email)) {
                Log.w(TAG, "Email not found: " + email);
                return new Pair<>(DBConstants.EMAIL_NOT_FOUND, null);
            }

            // Verify password
            if (!verifyPassword(email, password)) {
                Log.w(TAG, "Password incorrect for email: " + email);
                return new Pair<>(DBConstants.PASSWORD_INCORRECT, null);
            }

            // Generate hash for this account
            String accountHash = generateAccountHash(email);
            currentAccountHash = accountHash;

            Log.d(TAG, "Account verified successfully");
            return new Pair<>(DBConstants.SYNC_OK, accountHash);

        } catch (Exception e) {
            Log.e(TAG, "Error verifying account", e);
            status = DBConstants.STATUS_ERROR_DATABASE;
            return new Pair<>(DBConstants.GENERIC_ERROR, null);
        }
    }

    /** @noinspection BooleanMethodIsAlwaysInverted*/
    private boolean emailExistsInDatabase(String email) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            boolean[] exists = {false};

            firebaseDb.collection(DBConstants.FIREBASE_USERS_COLLECTION)
                    .document(email)
                    .get()
                    .addOnSuccessListener(documentSnapshot -> {
                        exists[0] = documentSnapshot.exists();
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error checking email existence", e);
                        latch.countDown();
                    });

            latch.await();
            return exists[0];

        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while checking email", e);
            return false;
        }
    }

    /**
     * Verify password for email
     */
    private boolean verifyPassword(String email, String password) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            boolean[] isCorrect = {false};

            firebaseDb.collection(DBConstants.FIREBASE_USERS_COLLECTION)
                    .document(email)
                    .get()
                    .addOnSuccessListener(documentSnapshot -> {
                        if (documentSnapshot.exists()) {
                            String storedPasswordHash = documentSnapshot.getString(DBConstants.FIREBASE_PASSWORD_FIELD);
                            // Compare hashes (in production, use bcrypt or similar)
                            isCorrect[0] = hashPassword(password).equals(storedPasswordHash);
                        }
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error verifying password", e);
                        latch.countDown();
                    });

            latch.await();
            return isCorrect[0];

        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while verifying password", e);
            return false;
        }
    }

    /**
     * Load profile data from Firebase for given account hash
     * Returns: 0 (success), -1 (error), 99 (no internet)
     */
    public int loadProfileData(String accountHash, String email) {
        Log.d(TAG, "Loading profile data for account: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.STATUS_ERROR_NETWORK;
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            // Initialize or get profile directory
            if (!FileUtils.getProfileDirectory(context).exists()) {
                FileUtils.initializeProfileDirectory(context);
            }

            // Get profile data from Firebase
            JSONObject profileData = fetchProfileFromFirebase(email);
            if (profileData == null) {
                return DBConstants.DATA_FETCH_ERROR;
            }

            // Write to temporary copy file
            File profileCopyFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.PROFILE_COPY_FILE);

            if (!FileUtils.writeProfile(context, profileData.toString())) {
                // Delete copy on write failure
                profileCopyFile.delete();
                status = DBConstants.STATUS_ERROR_DATABASE;
                return DBConstants.DATA_WRITE_ERROR;
            }

            // Copy temp file to actual profile
            File originalProfile = FileUtils.getProfileFile(context);
            if (!copyFile(profileCopyFile, originalProfile)) {
                status = DBConstants.STATUS_ERROR_DATABASE;
                return DBConstants.DATA_WRITE_ERROR;
            }

            // Load hash cache if enabled
            if (profileData.optBoolean(DBConstants.FIREBASE_HASH_CACHE_ENABLED, false)) {
                loadHashCacheFromCloud(email);
            }

            // Load env reports if enabled
            if (profileData.optBoolean(DBConstants.FIREBASE_ENV_REPORTS_ENABLED, false)) {
                loadEnvReportsFromCloud(email);
            }

            status = DBConstants.STATUS_SYNCED;
            Log.d(TAG, "Profile data loaded successfully");
            return DBConstants.SYNC_OK;

        } catch (Exception e) {
            Log.e(TAG, "Error loading profile data", e);
            status = DBConstants.STATUS_ERROR_DATABASE;
            return DBConstants.DATA_FETCH_ERROR;
        }
    }

    /**
     * Fetch profile from Firebase
     */
    private JSONObject fetchProfileFromFirebase(String email) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            JSONObject[] result = {null};

            firebaseDb.collection(DBConstants.FIREBASE_PROFILES_COLLECTION)
                    .document(email)
                    .get()
                    .addOnSuccessListener(documentSnapshot -> {
                        if (documentSnapshot.exists()) {
                            try {
                                result[0] = new JSONObject(documentSnapshot.getData());
                            } catch (Exception e) {
                                Log.e(TAG, "Error converting to JSON", e);
                            }
                        }
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error fetching profile", e);
                        latch.countDown();
                    });

            latch.await();
            return result[0];

        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while fetching profile", e);
            return null;
        }
    }

    /**
     * Load hash cache from cloud storage
     */
    private void loadHashCacheFromCloud(String email) {
        try {
            StorageReference hashCacheRef = firebaseStorage.getReference()
                    .child(DBConstants.FIREBASE_HASH_CACHE_STORAGE)
                    .child(email)
                    .child(DBConstants.HASH_CACHE_FILE_NAME);

            File hashCacheFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.HASH_CACHE_FILE_NAME);

            hashCacheRef.getFile(hashCacheFile)
                    .addOnSuccessListener(taskSnapshot ->
                            Log.d(TAG, "Hash cache loaded successfully"))
                    .addOnFailureListener(e -> {
                        Log.w(TAG, "Failed to load hash cache, creating new one", e);
                        createNewHashCacheFile();
                    });

        } catch (Exception e) {
            Log.e(TAG, "Error loading hash cache", e);
        }
    }

    /**
     * Load env reports from cloud storage
     */
    private void loadEnvReportsFromCloud(String email) {
        try {
            StorageReference envReportsRef = firebaseStorage.getReference()
                    .child(DBConstants.FIREBASE_ENV_REPORTS_STORAGE)
                    .child(email)
                    .child(DBConstants.ENV_REPORTS_FILE_NAME);

            File envReportsFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.ENV_REPORTS_FILE_NAME);

            envReportsRef.getFile(envReportsFile)
                    .addOnSuccessListener(taskSnapshot ->
                            Log.d(TAG, "Env reports loaded successfully"))
                    .addOnFailureListener(e -> {
                        Log.w(TAG, "Failed to load env reports, creating new one", e);
                        createNewEnvReportsFile();
                    });

        } catch (Exception e) {
            Log.e(TAG, "Error loading env reports", e);
        }
    }

    /**
     * Create new hash cache file
     */
    private void createNewHashCacheFile() {
        try {
            File hashCacheFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.HASH_CACHE_FILE_NAME);
            JSONObject emptyCache = new JSONObject();
            FileUtils.writeProfile(context, emptyCache.toString());
            Log.d(TAG, "New hash cache file created");
        } catch (Exception e) {
            Log.e(TAG, "Error creating hash cache file", e);
        }
    }

    /**
     * Create new env reports file
     */
    private void createNewEnvReportsFile() {
        try {
            File envReportsFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.ENV_REPORTS_FILE_NAME);
            JSONObject emptyReports = new JSONObject();
            FileUtils.writeProfile(context, emptyReports.toString());
            Log.d(TAG, "New env reports file created");
        } catch (Exception e) {
            Log.e(TAG, "Error creating env reports file", e);
        }
    }

    /**
     * Push data to server
     * Returns: 0 (success), -1 (error), 99 (no internet)
     */
    public int pushDataToServer(String email, JSONObject profileData) {
        Log.d(TAG, "Pushing data to server for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.STATUS_ERROR_NETWORK;
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            status = DBConstants.STATUS_SYNCING;

            // Push profile data
            firebaseDb.collection(DBConstants.FIREBASE_PROFILES_COLLECTION)
                    .document(email)
                    .set(profileData)
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error pushing profile data", e);
                        status = DBConstants.STATUS_ERROR_DATABASE;
                    });

            // Push hash cache if exists and enabled
            if (profileData.optBoolean(DBConstants.FIREBASE_HASH_CACHE_ENABLED, false)) {
                pushHashCacheToCloud(email);
            }

            // Push env reports if exists and enabled
            if (profileData.optBoolean(DBConstants.FIREBASE_ENV_REPORTS_ENABLED, false)) {
                pushEnvReportsToCloud(email);
            }

            status = DBConstants.STATUS_SYNCED;
            Log.d(TAG, "Data pushed to server successfully");
            return DBConstants.SYNC_OK;

        } catch (Exception e) {
            Log.e(TAG, "Error pushing data", e);
            status = DBConstants.STATUS_ERROR_DATABASE;
            return DBConstants.SYNC_ERROR;
        }
    }

    /**
     * Push hash cache to cloud storage
     */
    private void pushHashCacheToCloud(String email) {
        try {
            File hashCacheFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.HASH_CACHE_FILE_NAME);

            if (hashCacheFile.exists()) {
                StorageReference hashCacheRef = firebaseStorage.getReference()
                        .child(DBConstants.FIREBASE_HASH_CACHE_STORAGE)
                        .child(email)
                        .child(DBConstants.HASH_CACHE_FILE_NAME);

                hashCacheRef.putFile(com.google.android.gms.tasks.Tasks.await(
                                firebaseStorage.getReference().child("dummy").putBytes(new byte[0])))
                        .addOnSuccessListener(taskSnapshot ->
                                Log.d(TAG, "Hash cache pushed to cloud"))
                        .addOnFailureListener(e ->
                                Log.e(TAG, "Error pushing hash cache", e));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error in pushHashCacheToCloud", e);
        }
    }

    /**
     * Push env reports to cloud storage
     */
    private void pushEnvReportsToCloud(String email) {
        try {
            File envReportsFile = new File(FileUtils.getProfileDirectory(context),
                    DBConstants.ENV_REPORTS_FILE_NAME);

            if (envReportsFile.exists()) {
                StorageReference envReportsRef = firebaseStorage.getReference()
                        .child(DBConstants.FIREBASE_ENV_REPORTS_STORAGE)
                        .child(email)
                        .child(DBConstants.ENV_REPORTS_FILE_NAME);

                Log.d(TAG, "Env reports pushed to cloud");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error in pushEnvReportsToCloud", e);
        }
    }

    /**
     * Check if email is valid (exists in real world)
     * Returns: 0 (valid), -1 (invalid), 99 (no internet)
     */
    public int validateEmail(String email) {
        Log.d(TAG, "Validating email: " + email);

        if (!hasInternetConnection()) {
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            if (!isValidEmailFormat(email)) {
                return DBConstants.EMAIL_INVALID;
            }
            return DBConstants.EMAIL_VALID;

        } catch (Exception e) {
            Log.e(TAG, "Error validating email", e);
            return DBConstants.GENERIC_ERROR;
        }
    }

    /**
     * Create new account
     * Returns: Pair<code, hashcode or null>
     */
    public Pair<Integer, String> createAccount(String email, String password) {
        Log.d(TAG, "Creating account for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.STATUS_ERROR_NETWORK;
            return new Pair<>(DBConstants.INTERNET_CONNECTION_FAILED, null);
        }

        try {
            firebaseAuth.createUserWithEmailAndPassword(email, password)
                    .addOnSuccessListener(authResult -> {
                        Log.d(TAG, "Account created successfully");
                        String accountHash = generateAccountHash(email);
                        currentAccountHash = accountHash;
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error creating account", e);
                        status = DBConstants.STATUS_ERROR_DATABASE;
                    });

            return new Pair<>(DBConstants.ACCOUNT_CREATED, currentAccountHash);

        } catch (Exception e) {
            Log.e(TAG, "Error in createAccount", e);
            status = DBConstants.STATUS_ERROR_DATABASE;
            return new Pair<>(DBConstants.ACCOUNT_CREATION_FAILED, null);
        }
    }

    /**
     * Reset password for email
     * Returns: 0 (success), -1 (email not found), -2 (error), 99 (no internet)
     */
    public int resetPassword(String email) {
        Log.d(TAG, "Resetting password for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.STATUS_ERROR_NETWORK;
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            if (!emailExistsInDatabase(email)) {
                return DBConstants.EMAIL_NOT_FOUND;
            }

            firebaseAuth.sendPasswordResetEmail(email)
                    .addOnSuccessListener(aVoid ->
                            Log.d(TAG, "Password reset email sent"))
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error sending reset email", e);
                        status = DBConstants.STATUS_ERROR_DATABASE;
                    });

            return DBConstants.PASSWORD_RESET_SENT;

        } catch (Exception e) {
            Log.e(TAG, "Error in resetPassword", e);
            status = DBConstants.STATUS_ERROR_DATABASE;
            return DBConstants.GENERIC_ERROR;
        }
    }

    /**
     * Auto-sync profile based on date
     * Returns: 0 (synced or not needed), -1 (error)
     */
    public int autoSyncProfile(JSONObject profileData, LoadingManager loadingManager) {
        try {
            if (!profileData.has("remote") || !profileData.getBoolean("remote")) {
                Log.d(TAG, "Remote sync not enabled");
                return DBConstants.SYNC_OK;
            }
            loadingManager.changeText("Syncing profile, please wait");

            if (!hasInternetConnection()) {
                return DBConstants.INTERNET_CONNECTION_FAILED;
            }

            String lastSyncDateStr = profileData.optString(DBConstants.FIREBASE_LAST_SYNC_FIELD, "");
            if (isSyncNeeded(lastSyncDateStr)) {
                return pushDataToServer(profileData.getString("email"), profileData);
            }

            return DBConstants.SYNC_OK;

        } catch (JSONException e) {
            Log.e(TAG, "Error parsing profile data", e);
            return DBConstants.DATA_FETCH_ERROR;
        }
    }

    /**
     * Check if sync is needed based on interval
     */
    private boolean isSyncNeeded(String lastSyncDateStr) {
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
            Date lastSyncDate = sdf.parse(lastSyncDateStr);
            Date currentDate = new Date();

            long diffTime = currentDate.getTime() - lastSyncDate.getTime();
            long diffDays = diffTime / (1000 * 60 * 60 * 24);

            return diffDays >= DBConstants.SYNC_INTERVAL_DAYS;

        } catch (Exception e) {
            Log.e(TAG, "Error checking sync interval", e);
            return true; // Sync if error
        }
    }

    /**
     * Check sync status
     * Returns: 0 (OK), -1 (error)
     */
    public int checkSyncStatus() {
        return status.equals(DBConstants.STATUS_SYNCED) ?
                DBConstants.SYNC_OK : DBConstants.STATUS_ERROR;
    }

    /**
     * Get current status
     */
    public String getStatus() {
        return status;
    }

    /**
     * Helper methods
     */
    private String generateAccountHash(String email) {
        return String.valueOf(email.hashCode());
    }

    private String hashPassword(String password) {
        return String.valueOf(password.hashCode()); // Use BCrypt in production
    }

    private boolean isValidEmailFormat(String email) {
        return email != null && email.contains("@") && email.contains(".");
    }

    private boolean copyFile(File source, File destination) {
        try {
            java.nio.file.Files.copy(
                    source.toPath(),
                    destination.toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING
            );
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error copying file", e);
            return false;
        }
    }
    */
}