import React from 'react';

const Results = ({ results }) => {
  if (!results) return null;

  const formatMatrix = (matrix) => {
    return matrix.map((row, i) => (
      <div key={i} className="flex gap-4 justify-center font-mono text-sm">
        {row.map((val, j) => (
          <span key={j} className="w-32 text-right">
            {val.toFixed(4)}
          </span>
        ))}
      </div>
    ));
  };

  const formatArray = (arr) => {
    return arr.map((val, i) => (
      <span key={i} className="font-mono">
        {val.toFixed(6)}
        {i < arr.length - 1 ? ', ' : ''}
      </span>
    ));
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const downloadJSON = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'camera_calibration.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full max-w-4xl mx-auto mt-8">
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
            <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Calibration Results
          </h2>
          <button
            onClick={downloadJSON}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download JSON
          </button>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
            <div className="text-sm text-blue-600 font-semibold mb-1">Images Used</div>
            <div className="text-2xl font-bold text-blue-900">{results.num_images_used}</div>
          </div>
          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
            <div className="text-sm text-green-600 font-semibold mb-1">Mean Error</div>
            <div className="text-2xl font-bold text-green-900">
              {results.mean_reprojection_error?.toFixed(4) || 'N/A'}
            </div>
          </div>
          <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
            <div className="text-sm text-purple-600 font-semibold mb-1">Board Size</div>
            <div className="text-2xl font-bold text-purple-900">
              {results.pattern_size?.[0]} × {results.pattern_size?.[1]}
            </div>
          </div>
        </div>

        {/* Camera Matrix */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-xl font-semibold text-gray-800">Camera Matrix (K)</h3>
            <button
              onClick={() => copyToClipboard(JSON.stringify(results.camera_matrix))}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy
            </button>
          </div>
          <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
            {formatMatrix(results.camera_matrix)}
          </div>
          <div className="mt-3 text-sm text-gray-600 bg-blue-50 rounded-lg p-3">
            <strong>Parameters:</strong>
            <ul className="mt-2 space-y-1">
              <li>• f<sub>x</sub> (focal length X): <span className="font-mono">{results.camera_matrix[0][0].toFixed(2)}</span> pixels</li>
              <li>• f<sub>y</sub> (focal length Y): <span className="font-mono">{results.camera_matrix[1][1].toFixed(2)}</span> pixels</li>
              <li>• c<sub>x</sub> (principal point X): <span className="font-mono">{results.camera_matrix[0][2].toFixed(2)}</span> pixels</li>
              <li>• c<sub>y</sub> (principal point Y): <span className="font-mono">{results.camera_matrix[1][2].toFixed(2)}</span> pixels</li>
            </ul>
          </div>
        </div>

        {/* Distortion Coefficients */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-xl font-semibold text-gray-800">Distortion Coefficients</h3>
            <button
              onClick={() => copyToClipboard(JSON.stringify(results.distortion_coefficients))}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy
            </button>
          </div>
          <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
            <div className="font-mono text-sm text-center">
              [{formatArray(results.distortion_coefficients)}]
            </div>
          </div>
          <div className="mt-3 text-sm text-gray-600 bg-blue-50 rounded-lg p-3">
            <strong>Coefficients:</strong>
            <ul className="mt-2 space-y-1">
              <li>• k₁ (radial distortion): <span className="font-mono">{results.distortion_coefficients[0].toFixed(6)}</span></li>
              <li>• k₂ (radial distortion): <span className="font-mono">{results.distortion_coefficients[1].toFixed(6)}</span></li>
              <li>• p₁ (tangential distortion): <span className="font-mono">{results.distortion_coefficients[2].toFixed(6)}</span></li>
              <li>• p₂ (tangential distortion): <span className="font-mono">{results.distortion_coefficients[3].toFixed(6)}</span></li>
              <li>• k₃ (radial distortion): <span className="font-mono">{results.distortion_coefficients[4].toFixed(6)}</span></li>
            </ul>
          </div>
        </div>

        {/* Info box */}
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200">
          <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
                How to Use These Results
              </h4>
              <p className="text-sm text-gray-700 leading-relaxed">
                Use the camera matrix and distortion coefficients in OpenCV to correct images 
                with the <code className="bg-white px-2 py-0.5 rounded">cv2.undistort()</code> function. 
                A low mean error (less than 1.0) indicates good calibration quality.
              </p>
        </div>
      </div>
    </div>
  );
};

export default Results;

