package com.visionassist.appspace.database;

import android.content.Context;
import android.util.Log;
import android.util.Pair;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;
import com.visionassist.appspace.ExceptionVisionAssist;
import com.visionassist.appspace.jetpack.managers.LoadingManager;
import com.visionassist.appspace.utils.Constants;
import com.visionassist.appspace.utils.FileUtils;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

public class DBManager {
    private static final String TAG = "DBManager";

    private Context context;
    private FirebaseFirestore firebaseDb;
    private final FirebaseAuth auth;
    private int status = DBConstants.STATUS_INITIALIZED;

    public DBManager(Context context) {
        this.context = context;
        this.firebaseDb = FirebaseFirestore.getInstance();
        this.auth = FirebaseAuth.getInstance();
        Log.d(TAG, "DBManager initialized, FirebaseAuth(auth) and FirebaseFirestore(data persistence)");
    }

    private boolean hasInternetConnection() {
        return NetworkUtils.isNetworkConnected(context);
    }

    public int verifyAccount(String email, String password) {
        Log.d(TAG, "Verifying account for email: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.INTERNET_CONNECTION_FAILED;
            Log.w(TAG, "No internet connection");
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            if (!emailExistsInDatabase(email)) {
                Log.w(TAG, "Email not found: " + email);
                return DBConstants.EMAIL_NOT_FOUND;
            }

            if (!verifyPassword(email, password)) {
                Log.w(TAG, "Password incorrect for email: " + email);
                return DBConstants.PASSWORD_INCORRECT;
            }

            Log.d(TAG, "Account verified successfully");
            return DBConstants.SYNC_OK;

        } catch (Exception e) {
            Log.e(TAG, "Error verifying account", e);
            status = DBConstants.GENERIC_ERROR;
            return DBConstants.GENERIC_ERROR;
        }
    }

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

    private boolean verifyPassword(String email, String password) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            AtomicBoolean success = new AtomicBoolean(false);

            auth.signInWithEmailAndPassword(email, password)
                    .addOnCompleteListener(authTask -> {
                        if (authTask.isSuccessful()) {
                            Log.d(TAG, "Password valid, verified from Realtime Database");
                            success.set(true);
                            latch.countDown();
                        } else {
                            Log.d(TAG, "Password invalid, verified from Realtime Database");
                            latch.countDown();
                        }
                    });

            latch.await();

            return success.get();
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while verifying password", e);
            return false;
        }
    }

    public int validateEmail(String email) {
        Log.d(TAG, "Validating email: " + email);

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

    public Pair<String, Integer> createAccount(String email, String password) {
        Log.d(TAG, "Creating account for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.INTERNET_CONNECTION_FAILED;
            return null;
        }

        try {
            CountDownLatch latch = new CountDownLatch(1);
            AtomicBoolean success = new AtomicBoolean(false);
            String[] userID={"salut"};

            auth.createUserWithEmailAndPassword(email, password)
                    .addOnCompleteListener(authTask -> {
                        if (authTask.isSuccessful()) {
                            FirebaseUser user = auth.getCurrentUser();
                            if (user != null) {
                                userID[0] = user.getUid();
                                Log.d(TAG, "User account created successfully in Realtime Database\nUID: "+user);
                                success.set(true);
                                latch.countDown();
                            }
                            else
                                latch.countDown();
                        } else {
                            Log.e(TAG, "Error creating user account in Realtime Database");
                            latch.countDown();
                        }
                    });

            latch.await();

            if (success.get()) {
                status = DBConstants.ACCOUNT_CREATED;
            } else {
                status = DBConstants.ACCOUNT_CREATION_FAILED;
            }
            return new Pair<>(userID[0], status);
        } catch (Exception e) {
            Log.e(TAG, "Error in createAccount", e);
            status = DBConstants.ACCOUNT_CREATION_FAILED;
            return null;
        }
    }

    public void syncProfile(JSONObject profileData) throws JSONException {
        try {
            if (!hasInternetConnection()) {
                status = DBConstants.INTERNET_CONNECTION_FAILED;
                return;
            }

            String email = profileData.getString("email");
            pushProfile(email, profileData);
        } catch (JSONException e) {
            Log.e(TAG, "Error parsing profile data", e);
            throw e;
        }
    }

    public void pushProfile(String email, JSONObject profileData) {
        Log.d(TAG, "Pushing profile to Firestore for: " + email);

        try {
            // Update last_sync_date in profile
            String currentDate = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
                    .format(new Date());
            profileData.put(DBConstants.FIREBASE_LAST_SYNC_FIELD, currentDate);

            // Convert JSONObject to Map for Firestore
            Map<String, Object> profileMap = jsonToMap(profileData);

            // Push to Firestore: users/{email}/
            CountDownLatch latch = new CountDownLatch(1);
            AtomicBoolean success = new AtomicBoolean(false);

            firebaseDb.collection(DBConstants.FIREBASE_USERS_COLLECTION)
                    .document(email)
                    .set(profileMap)
                    .addOnSuccessListener(aVoid -> {
                        Log.d(TAG, "Profile pushed to Firestore successfully");
                        success.set(true);
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error pushing profile to Firestore", e);
                        latch.countDown();
                    });

            latch.await();

            if (!success.get()) {
                status = DBConstants.SYNC_ERROR;
                return;
            }

            // Update local profile.json with new last_sync_date
            FileUtils.writeProfileFile(profileData.toString(), Constants.PROFILE_FILE_NAME);
            status = DBConstants.SYNC_OK;
            Log.d(TAG, "Profile push completed successfully (Firestore only)");
        } catch (Exception e) {
            Log.e(TAG, "Error pushing profile", e);
            status = DBConstants.SYNC_ERROR;
        }
    }

    public Pair<Integer, JSONObject> pullProfile(String email) {
        Log.d(TAG, "Pulling profile from Firestore for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.INTERNET_CONNECTION_FAILED;
            return new Pair<>(status, null);
        }

        try {
            // Step 1: Fetch profile from Firestore
            JSONObject profileData = fetchProfileFromFirestore(email);
            if (profileData == null) {
                Log.e(TAG, "Failed to fetch profile from Firestore");
                status = DBConstants.DATA_FETCH_ERROR;
                return new Pair<>(status, null);
            }

            FileUtils.createProfileDirFile(DBConstants.PROFILE_COPY_FILE);

            if (!FileUtils.writeProfileFile(profileData.toString(), DBConstants.PROFILE_COPY_FILE)) {
                Log.e(TAG, "Failed to write profile_copy.json");
                status = DBConstants.DATA_WRITE_ERROR;
                return new Pair<>(status, null);
            }

            // Step 2: Create empty files for hash_cache and env_reports if enabled
            boolean hashCacheEnabled = profileData.optString("hash_caching", "none").equals("heavy")
                    || profileData.optString("hash_caching", "none").equals("light");
            boolean envReportsEnabled = profileData.optBoolean("env_reports", false);

            boolean filesReady = true;

            // Create empty hash_cache file if enabled
            if (hashCacheEnabled) {
                if (!FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)) {
                    Log.e(TAG, "Failed to create hash_cache file");
                    filesReady = false;
                }
            }

            // Create empty env_reports file if enabled
            if (envReportsEnabled) {
                if (!FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)) {
                    Log.e(TAG, "Failed to create env_reports file");
                    filesReady = false;
                }
            }

            File originalProfileFile = FileUtils.getProfileFile(context);
            File profileCopyFile = new File(FileUtils.getProfileDirectory(context), DBConstants.PROFILE_COPY_FILE);
            // Step 3: All files ready - NOW replace original profile.json
            if (filesReady) {
                if (!copyFile(profileCopyFile, originalProfileFile)) {
                    Log.e(TAG, "Failed to copy profile_copy.json to profile.json");
                    status = DBConstants.DATA_WRITE_ERROR;
                    return new Pair<>(status, null);
                }
                Log.d(TAG, "Profile successfully copied to profile.json");
                if (profileCopyFile.delete())
                    Log.d(TAG, "Deletion failed of profile.json");
                else
                    Log.d(TAG, "Deletion successfully of profile.json");
                status = DBConstants.SYNC_OK;
                return new Pair<>(status, profileData);
            } else {
                Log.e(TAG, "Failed to create required files, aborting profile replacement");
                if (profileCopyFile.delete())
                    Log.d(TAG, "Deletion failed of profile.json");
                else
                    Log.d(TAG, "Deletion successfully of profile.json");
                status = DBConstants.DATA_WRITE_ERROR;
                return new Pair<>(status, null);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error pulling profile", e);
            status = DBConstants.DATA_FETCH_ERROR;
            return new Pair<>(status, null);
        }
    }

    private JSONObject fetchProfileFromFirestore(String email) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            JSONObject[] result = {null};

            firebaseDb.collection(DBConstants.FIREBASE_USERS_COLLECTION)
                    .document(email)
                    .get()
                    .addOnSuccessListener(documentSnapshot -> {
                        if (documentSnapshot.exists()) {
                            try {
                                Map<String, Object> data = documentSnapshot.getData();
                                if (data != null) {
                                    result[0] = new JSONObject(data);
                                    Log.d(TAG, "Profile fetched successfully from Firestore");
                                }
                            } catch (Exception e) {
                                Log.e(TAG, "Error converting document to JSON", e);
                            }
                        } else {
                            Log.w(TAG, "Profile document does not exist in Firestore");
                        }
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Error fetching profile from Firestore", e);
                        latch.countDown();
                    });

            latch.await();
            return result[0];
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while fetching profile", e);
            return null;
        }
    }

    public int resetPassword(String email) {
        Log.d(TAG, "Resetting password for: " + email);

        if (!hasInternetConnection()) {
            status = DBConstants.INTERNET_CONNECTION_FAILED;
            return DBConstants.INTERNET_CONNECTION_FAILED;
        }

        try {
            if (!emailExistsInDatabase(email)) {
                return DBConstants.EMAIL_NOT_FOUND;
            }

            CountDownLatch latch = new CountDownLatch(1);
            AtomicBoolean success = new AtomicBoolean(false);

            Log.d(TAG, "Sending password reset email to: " + email);
            auth.sendPasswordResetEmail(email)
                    .addOnCompleteListener(task -> {
                        if (task.isSuccessful()) {
                            Log.d(TAG, "Password reset email sent successfully");
                            success.set(true);
                            latch.countDown();
                        } else {
                            Log.e(TAG, "Failed to send password reset email");
                            latch.countDown();
                        }
                    });

            latch.await();

            return success.get()?DBConstants.PASSWORD_RESET_SENT:DBConstants.GENERIC_ERROR;
        } catch (Exception e) {
            Log.e(TAG, "Error in resetPassword", e);
            status = DBConstants.GENERIC_ERROR;
            return DBConstants.GENERIC_ERROR;
        }
    }

    public boolean isRemoteProfile(JSONObject profileData, LoadingManager loadingManager) throws ExceptionVisionAssist {
        try {
            if (!profileData.getBoolean("remote")) {
                Log.d(TAG, "Remote sync not enabled");
                status = DBConstants.SYNC_NOT_NEEDED;
                return false;
            }

            String lastSyncDateStr = profileData.getString("last_sync_date");
            if (!isSyncNeeded(lastSyncDateStr)) {
                Log.d(TAG, "Sync not needed yet");
                status = DBConstants.SYNC_OK;
                return false;
            }

            return true;
        } catch (JSONException e) {
            Log.e(TAG, "Error checking remote profile", e);
            throw new ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, loadingManager);
        }
    }

    private boolean isSyncNeeded(String lastSyncDateStr) {
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
            Date lastSyncDate = sdf.parse(lastSyncDateStr);

            if (lastSyncDate == null) {
                return true;
            }

            // Use Calendar to calculate days difference properly
            Calendar lastSync = Calendar.getInstance();
            lastSync.setTime(lastSyncDate);
            lastSync.set(Calendar.HOUR_OF_DAY, 0);
            lastSync.set(Calendar.MINUTE, 0);
            lastSync.set(Calendar.SECOND, 0);
            lastSync.set(Calendar.MILLISECOND, 0);

            Calendar current = Calendar.getInstance();
            current.set(Calendar.HOUR_OF_DAY, 0);
            current.set(Calendar.MINUTE, 0);
            current.set(Calendar.SECOND, 0);
            current.set(Calendar.MILLISECOND, 0);

            long diffTime = current.getTimeInMillis() - lastSync.getTimeInMillis();
            long diffDays = diffTime / (1000 * 60 * 60 * 24);

            Log.d(TAG, "Days since last sync: " + diffDays);
            return diffDays >= DBConstants.SYNC_INTERVAL_DAYS;

        } catch (Exception e) {
            Log.e(TAG, "Error checking sync interval", e);
            return true; // Sync if error
        }
    }

    private boolean copyFile(File source, File destination) {
        try (FileInputStream fis = new FileInputStream(source);
             FileOutputStream fos = new FileOutputStream(destination)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = fis.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Error copying file", e);
            return false;
        }
    }

    private Map<String, Object> jsonToMap(JSONObject json) throws JSONException {
        Map<String, Object> map = new HashMap<>();
        java.util.Iterator<String> keys = json.keys();

        while (keys.hasNext()) {
            String key = keys.next();
            Object value = json.get(key);

            if (value instanceof JSONObject) {
                value = jsonToMap((JSONObject) value);
            }

            map.put(key, value);
        }

        return map;
    }

    private boolean isValidEmailFormat(String email) {
        return email != null && email.matches(".+@.+\\..+");
    }

    public int getStatus() {
        return status;
    }

    public int getStatusOverview() {
        return switch (status) {
            case DBConstants.SYNC_OK -> 1;
            case DBConstants.INTERNET_CONNECTION_FAILED, DBConstants.SYNC_ERROR -> 2;
            default -> 0;
        };
    }
}