package com.nmsolver.linalg;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.nmsolver.ScsSettings;

import lombok.AllArgsConstructor;
import lombok.NonNull;

@AllArgsConstructor
public class DCSCMatrix {
	public int nRow; // m
	public int nCol; // n
	public @NonNull int[] colPtr; // p, nCol + 1
	public @NonNull int[] rowIdx; // i, nz
	public @NonNull double[] data; // x, nz
	private static ExecutorService e = Executors.newFixedThreadPool(ScsSettings.THREAD_COUNT);

	public double[] diag() {
		var size = Math.min(nCol, nRow);
		var ret = new double[size];
		for (int i = 0; i < nCol; ++i) {
			for (int k = colPtr[i]; k < colPtr[i + 1]; ++k) {
				if (rowIdx[k] == i) {
					ret[i] = data[k];
					break;
				}
			}
		}
		return ret;
	}

	private static void cumsum(int[] p, int[] c, int n) {
		int nz = 0;
		for (int i = 0; i < n; i++) {
			p[i] = nz;
			nz += c[i];
			c[i] = p[i];
		}
		p[n] = nz;
	}

	public DCSCMatrix transpose() {
		var colPtrT = new int[nRow + 1];
		var rowIdxT = new int[colPtr[nCol]];
		var dataT = new double[colPtr[nCol]];

		// row counts
		var rc = new int[nRow + 1];
		for (int i = 0; i < colPtr[nCol]; ++i) {
			rc[rowIdx[i]]++;
		}
		cumsum(colPtrT, rc, nRow);

		for (int j = 0; j < nCol; ++j) {
			var c1 = colPtr[j];
			var c2 = colPtr[j + 1];
			for (int i = c1; i < c2; ++i) {
				var q = rc[rowIdx[i]];
				rowIdxT[q] = j;
				dataT[q] = data[i];
				rc[rowIdx[i]]++;
			}
		}
		return new DCSCMatrix(nCol, nRow, colPtrT, rowIdxT, dataT);
	}

	@Override
	public String toString() {
		var b = new StringBuilder();
		b.append("nRow: " + nRow + "\tnCol: " + nCol + "\n");
		b.append("colPtr: " + Arrays.toString(colPtr) + "\n");
		b.append("rowIdx: " + Arrays.toString(rowIdx) + "\n");
		b.append("data: " + Arrays.toString(data));
		return b.toString();
	}

	/* y += A'x */
	public static void accumByATx(final DCSCMatrix A, final DVector x, DVector y) {
		if (ScsSettings.PARALLEL) {
			accumByATxParallel(A, x, y);
			return;
		}
		for (int j = 0; j < A.nCol; ++j) {
			for (int p = A.colPtr[j]; p < A.colPtr[j + 1]; ++p) {
				y.data[j] += A.data[p] * x.data[A.rowIdx[p]];
			}
		}
	}

	private static void accumByATxParallel(final DCSCMatrix A, final DVector x, DVector y) {
		var doneSig = new CountDownLatch(ScsSettings.THREAD_COUNT);
		for (int i = 0; i < ScsSettings.THREAD_COUNT; ++i) {
			e.submit(new AccumATxWorker(doneSig, A, x, y, i));
		}
		try {
			doneSig.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	/* y += Ax, slow due to discontinuous memory accesses */
	public static void accumByAx(final DCSCMatrix A, final DVector x, DVector y) {
		for (int j = 0; j < A.nCol; ++j) {
			for (int p = A.colPtr[j]; p < A.colPtr[j + 1]; ++p) {
				var i = A.rowIdx[p];
				y.data[i] += A.data[p] * x.data[j];
			}
		}
	}

	/* y += Px, P is positive semi-definite, and upper triangular */
	public static void accumByPx(final DCSCMatrix P, final DVector x, DVector y) {
		for (int j = 0; j < P.nCol; ++j) {
			for (int p = P.colPtr[j]; p < P.colPtr[j + 1]; ++p) {
				var i = P.rowIdx[p];
				if (i != j) {
					y.data[i] += P.data[p] * x.data[j];
				}
			}
		}
		/* y += P_lower x */
		accumByATx(P, x, y);
	}

	static class AccumATxWorker implements Runnable {
	private CountDownLatch doneSig;
	final DCSCMatrix A;
	final DVector x;
	DVector y;
	int start, end;

	public AccumATxWorker(
			CountDownLatch doneSig,
			DCSCMatrix A,
			DVector x,
			DVector y, int rank) {
		this.doneSig = doneSig;
		this.A = A;
		this.x = x;
		this.y = y;
		int count = A.nCol / ScsSettings.THREAD_COUNT;

		this.start = count * rank;
		this.end = (rank == ScsSettings.THREAD_COUNT - 1) ? A.nCol : (start + count);
	}

	@Override
	public void run() {
		for (int j = start; j < end; ++j) {
			for (int p = A.colPtr[j]; p < A.colPtr[j + 1]; ++p) {
				y.data[j] += A.data[p] * x.data[A.rowIdx[p]];
			}
		}
		doneSig.countDown();
	}
}
}

