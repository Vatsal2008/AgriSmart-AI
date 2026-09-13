package com.agrismart.offline;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * Lends the captured leaf photo (cache/share/*.jpg) to Google Lens for a picture search, read-only and only
 * through the temporary permission granted with the share intent. No AndroidX needed.
 */
public class PhotoProvider extends ContentProvider {
    static final String AUTHORITY = "com.agrismart.offline.photos";

    static File shareDir(android.content.Context context) {
        return new File(context.getCacheDir(), "share");
    }

    private File file(Uri uri) throws FileNotFoundException {
        String name = uri.getLastPathSegment();
        if (name == null || name.contains("/") || name.contains("..")) throw new FileNotFoundException(String.valueOf(uri));
        File f = new File(shareDir(getContext()), name);
        if (!f.isFile()) throw new FileNotFoundException(String.valueOf(uri));
        return f;
    }

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        return ParcelFileDescriptor.open(file(uri), ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override
    public String getType(Uri uri) {
        return "image/jpeg";
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] args, String sortOrder) {
        MatrixCursor c = new MatrixCursor(new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});
        try {
            File f = file(uri);
            c.addRow(new Object[]{f.getName(), f.length()});
        } catch (FileNotFoundException ignored) {
            // empty cursor
        }
        return c;
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        return null;
    }

    @Override
    public int delete(Uri uri, String selection, String[] args) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] args) {
        return 0;
    }
}
