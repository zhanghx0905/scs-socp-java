package com.nmsolver;

import java.io.FileReader;
import java.io.IOException;

import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DMatrixSparseTriplet;
import org.ejml.ops.MatrixIO;
import org.ejml.ops.DConvertMatrixStruct;

import com.nmsolver.linalg.DCSCMatrix;
import com.nmsolver.linalg.DVector;

import us.hebi.matlab.mat.ejml.Mat5Ejml;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.types.MatFile;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;
import us.hebi.matlab.mat.types.Sparse;

public class ScsIO {
    private static DMatrixSparseTriplet readMtx(String path) throws IOException {
        var reader = new FileReader(path);
        var ret = MatrixIO.loadMatrixMarketD(reader);
        reader.close();
        return ret;
    }

    public static DCSCMatrix readMtxCSC(String path) throws IOException {
        DMatrixSparseCSC dst = null;
        dst = DConvertMatrixStruct.convert(readMtx(path), dst);
        assert dst != null;
        return new DCSCMatrix(dst.numRows, dst.numCols,
                dst.col_idx, dst.nz_rows, dst.nz_values);
    }

    public static DVector readMtxVector(String path) throws IOException {
        DMatrixRMaj dst = null;
        dst = DConvertMatrixStruct.convert(readMtx(path), dst);
        assert dst != null;
        return new DVector(dst.data);
    }

    public static ScsData readMat(String path) throws IOException {
        ScsData ret = null;
        try (Source source = Sources.openFile(path)) {
            MatFile mat = Mat5.newReader(source).readMat();
            Sparse A = mat.getSparse("A");
            Sparse b = mat.getSparse("b");
            Sparse c = mat.getSparse("c");
            DMatrixSparseCSC Adst = Mat5Ejml.convert(A);
            DMatrixSparseCSC btmp = Mat5Ejml.convert(b);
            DMatrixSparseCSC ctmp = Mat5Ejml.convert(c);
            DMatrixRMaj dst = null;
            DMatrixRMaj bdst = DConvertMatrixStruct.convert(btmp, dst);
            DMatrixRMaj cdst = DConvertMatrixStruct.convert(ctmp, dst);
            var Aout = new DCSCMatrix(Adst.numRows, Adst.numCols,
                    Adst.col_idx, Adst.nz_rows, Adst.nz_values);
            var bout = new DVector(bdst.data);
            var cout = new DVector(cdst.data);
            ret = new ScsData(Aout, bout, cout);
        }
        return ret;
    }
}
