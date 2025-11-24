  import React, { useState } from 'react';
  import axios from 'axios';
  import FileUpload from './components/FileUpload';
  import CameraCapture from './components/CameraCapture';
  import Results from './components/Results';

  const API_URL = 'http://localhost:8000';

  const mapFilesToEntries = (files) =>
    files.map((file, index) => ({
      id: `${file.name || 'file'}-${file.size || 'size'}-${file.lastModified || Date.now()}-${index}-${Math.random()
        .toString(36)
        .slice(2, 8)}`,
      file,
    }));

  // Separate component for image preview item to isolate URL management
  const ImagePreviewItem = ({ file, index, onRemove }) => {
    const [url, setUrl] = React.useState(() => URL.createObjectURL(file));
    
    React.useEffect(() => {
      // Create URL when component mounts
      const objectUrl = URL.createObjectURL(file);
      setUrl(objectUrl);
      
      // Cleanup: revoke URL when component unmounts or file changes
      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    }, [file]);

    return (
      <div className="relative group cursor-pointer">
        <img
          src={url}
          alt={`Preview ${index + 1}`}
          className="w-full h-24 object-cover rounded-lg border-2 border-gray-200 transition-all duration-200 group-hover:brightness-50"
          onError={(e) => {
            console.error('Failed to load image preview:', file.name);
          }}
        />
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onRemove();
          }}
          className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 shadow-md z-10"
          title="Remove image"
          type="button"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <div className="absolute bottom-1 left-1 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded z-10">
          {index + 1}
        </div>
      </div>
    );
  };

  function App() {
    const [mode, setMode] = useState(null); // 'upload', 'camera', 'self-upload', 'self-camera'
    const [calibrationType, setCalibrationType] = useState(null); // 'chessboard' or 'self'
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    
    // Calibration parameters (for chessboard method)
    const [patternWidth, setPatternWidth] = useState(8);
    const [patternHeight, setPatternHeight] = useState(6);
    const [squareSize, setSquareSize] = useState(25);

    const handleImagesSelected = (selectedFiles) => {
      const entries = mapFilesToEntries(selectedFiles);
      setImages(entries);
      setResults(null);
      setError(null);
    };

    // Remove image by id for deterministic behavior
    const handleRemoveImage = (imageId) => {
      const filter = images.filter(image => image.id !== imageId);
      setImages(filter);
      setResults(null);
      setError(null);
    };

    const handleClearAll = () => {
      setImages([]);
      setResults(null);
      setError(null);
    };

    const handleCalibrate = async () => {
    const isSelfCalibration = calibrationType === 'self';
    const minImages = isSelfCalibration ? 10 : 5;
    const maxImages = isSelfCalibration ? 500 : 50;
      
      if (images.length < minImages) {
        alert(`Please ${isSelfCalibration ? 'upload' : 'capture'} at least ${minImages} images`);
        return;
      }
    if (images.length > maxImages) {
      alert(`Please ${isSelfCalibration ? 'upload' : 'capture'} no more than ${maxImages} images for this method`);
      return;
    }

      setLoading(true);
      setError(null);
      setResults(null);

      try {
        const formData = new FormData();
        images.forEach(({ file }) => {
          formData.append('files', file);
        });

        let response;
        if (isSelfCalibration) {
          // Self-calibration endpoint
          response = await axios.post(
            `${API_URL}/self-calibrate`,
            formData,
            {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            }
          );
        } else {
          // Chessboard calibration endpoint
          response = await axios.post(
            `${API_URL}/calibrate?pattern_width=${patternWidth}&pattern_height=${patternHeight}&square_size=${squareSize}`,
            formData,
            {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            }
          );
        }

        setResults(response.data);
      } catch (err) {
        console.error('Calibration error:', err);
        setError(
          err.response?.data?.detail || 
          (calibrationType === 'self' 
            ? 'Self-calibration failed. Ensure images show a scene from different angles with sufficient features.'
            : 'Calibration failed. Please ensure images contain a visible chessboard pattern.')
        );
      } finally {
        setLoading(false);
      }
    };

    const handleReset = () => {
      setMode(null);
      setCalibrationType(null);
      setImages([]);
      setResults(null);
      setError(null);
    };

    const handleBack = () => {
      if (results) {
        // From results -> back to start
        handleReset();
      } else if (mode) {
        // From upload/camera mode -> back to mode selection
        setMode(null);
        setImages([]);
        setError(null);
      } else if (calibrationType) {
        // From mode selection -> back to calibration type selection
        setCalibrationType(null);
      }
    };

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
        {/* Header */}
        <header className="bg-white shadow-md">
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <svg className="w-10 h-10 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <div>
                  <h1 className="text-3xl font-bold text-gray-800">Camera Calibration</h1>
                </div>
              </div>
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 py-8">
          {/* Navigation buttons on page */}
          {(calibrationType || mode || results) && (
            <div className="max-w-4xl mx-auto mb-6">
              <div className="flex items-center justify-between">
                <button
                  onClick={handleBack}
                  className="px-4 py-2 bg-white hover:bg-gray-50 border-2 border-gray-300 hover:border-blue-400 rounded-lg transition-all font-medium flex items-center gap-2 text-gray-700 hover:text-blue-600 shadow-sm"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  Back
                </button>
                
                {(mode || results) && (
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 border-2 border-gray-300 rounded-lg transition-colors font-medium flex items-center gap-2 text-gray-600 hover:text-gray-800"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Start Over
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Progress Breadcrumbs */}
          {(calibrationType || mode) && !results && (
            <div className="max-w-4xl mx-auto mb-8">
              <div className="flex items-center justify-center gap-2 text-sm">
                {/* Step 1 */}
                <div className={`flex items-center gap-2 ${calibrationType ? 'text-green-600' : 'text-blue-600 font-semibold'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${calibrationType ? 'bg-green-100' : 'bg-blue-100'}`}>
                    {calibrationType ? '✓' : '1'}
                  </div>
                  <span>Choose Method</span>
                </div>
                
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                
                {/* Step 2 */}
                <div className={`flex items-center gap-2 ${mode ? 'text-green-600' : calibrationType ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${mode ? 'bg-green-100' : calibrationType ? 'bg-blue-100' : 'bg-gray-100'}`}>
                    {mode ? '✓' : '2'}
                  </div>
                  <span>Input Method</span>
                </div>
                
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                
                {/* Step 3 */}
                <div className={`flex items-center gap-2 ${mode ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${mode ? 'bg-blue-100' : 'bg-gray-100'}`}>
                    3
                  </div>
                  <span>Calibrate</span>
                </div>
              </div>
            </div>
          )}

          {/* Calibration Type Selection */}
          {!calibrationType && (
            <div className="max-w-5xl mx-auto">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-3">
                  Choose Calibration Method
                </h2>
                <p className="text-gray-600">
                  Select the method that best fits your needs
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Chessboard Calibration */}
                <button
                  onClick={() => setCalibrationType('chessboard')}
                  className="group p-8 bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-primary-500"
                >
                  <div className="flex flex-col items-center">
                    <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center mb-4 group-hover:bg-primary-200 transition-colors">
                      <svg className="w-10 h-10 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      Chessboard Calibration
                    </h3>
                    <p className="text-gray-600 text-center mb-3">
                      Traditional method using a chessboard pattern
                    </p>
                    <span className="inline-block px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
                      More Accurate
                    </span>
                  </div>
                </button>

                {/* Self-Calibration */}
                <button
                  onClick={() => setCalibrationType('self')}
                  className="group p-8 bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-blue-500"
                >
                  <div className="flex flex-col items-center">
                    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
                      <svg className="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      Self-Calibration
                    </h3>
                    <p className="text-gray-600 text-center mb-3">
                      No pattern needed - just photos of any scene
                    </p>
                    <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded-full">
                      No Setup Required
                    </span>
                  </div>
                </button>
              </div>
            </div>
          )}

          {/* Mode Selection (after calibration type chosen) */}
          {calibrationType && !mode && (
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-3">
                  Choose Image Input Method
                </h2>
                <p className="text-gray-600">
                  {calibrationType === 'chessboard' 
                    ? "You'll need a chessboard pattern for camera calibration"
                    : "Take photos of any scene from different angles"}
                </p>
              </div>

              {/* Chessboard PDF Download */}
              {calibrationType === 'chessboard' && (
                <div className="mb-8 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl p-6 border-2 border-blue-200">
                  <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-start gap-3 flex-1">
                      <svg className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div>
                        <h3 className="text-lg font-semibold text-gray-800 mb-1">
                          Need a Chessboard Pattern?
                        </h3>
                        <p className="text-sm text-gray-700">
                          Don't have a calibration chessboard? Download and print our ready-to-use pattern (A4, 9×7, 25mm squares).
                        </p>
                      </div>
                    </div>
                    <a
                      href="/calib.io_checker_210x297_9x7_25.pdf"
                      download
                      className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md hover:shadow-lg transition-all duration-200 whitespace-nowrap"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Download PDF
                    </a>
                  </div>
                </div>
              )}

              <div className="grid md:grid-cols-2 gap-6">
                {/* Upload Mode */}
                <button
                  onClick={() => setMode(calibrationType === 'self' ? 'self-upload' : 'upload')}
                  className="group p-8 bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-primary-500"
                >
                  <div className="flex flex-col items-center">
                    <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center mb-4 group-hover:bg-primary-200 transition-colors">
                      <svg className="w-10 h-10 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      Upload Photos
                    </h3>
                    <p className="text-gray-600 text-center">
                      {calibrationType === 'chessboard' 
                        ? "You already have chessboard images ready"
                        : "You already have scene images ready"}
                    </p>
                  </div>
                </button>

                {/* Camera Mode */}
                <button
                  onClick={() => setMode(calibrationType === 'self' ? 'self-camera' : 'camera')}
                  className="group p-8 bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-blue-500"
                >
                  <div className="flex flex-col items-center">
                    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
                      <svg className="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      Use Camera
                    </h3>
                    <p className="text-gray-600 text-center">
                      Capture images directly from your device camera
                    </p>
                  </div>
                </button>
              </div>

              {/* Info Section */}
              <div className="mt-12 bg-blue-50 rounded-xl p-6 border border-blue-200">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Capture Tips
                </h3>
                {calibrationType === 'chessboard' ? (
                  <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Take 15-20 photos of the chessboard at different angles</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Ensure the entire board is visible in each photo</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Vary the distance and orientation of the board</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Avoid blurred images and reflections</span>
                    </li>
                  </ul>
                ) : (
                  <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Take 100+ photos of any textured scene from different viewpoints (max 500)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Move the camera around (rotate and translate)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Ensure enough overlap between consecutive images</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>Avoid blurry images and pure textureless surfaces</span>
                    </li>
                  </ul>
                )}
              </div>
            </div>
          )}

          {/* Calibration Parameters (only for chessboard) */}
          {mode && !results && calibrationType === 'chessboard' && (
            <div className="max-w-4xl mx-auto mb-8">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Chessboard Parameters</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Width (corners)
                    </label>
                    <input
                      type="number"
                      value={patternWidth}
                      onChange={(e) => setPatternWidth(parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      min="3"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Height (corners)
                    </label>
                    <input
                      type="number"
                      value={patternHeight}
                      onChange={(e) => setPatternHeight(parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      min="3"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Square size (mm)
                    </label>
                    <input
                      type="number"
                      value={squareSize}
                      onChange={(e) => setSquareSize(parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      min="1"
                    />
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-3">
                  Corners = inner corners of the chessboard (not squares!). For example, an 8×8 board has 7×7 corners.
                </p>
              </div>
            </div>
          )}

          {/* Upload Mode */}
          {(mode === 'upload' || mode === 'self-upload') && !results && (
            <div className="max-w-4xl mx-auto">
              <div className="bg-white rounded-xl shadow-lg p-8">
                <FileUpload 
                  onImagesSelected={handleImagesSelected}
                currentImages={images.map(image => image.file)}
                  label={calibrationType === 'self' ? 'Upload scene images' : 'Upload chessboard images'}
                  showPreview={false}
                />
                
                {images.length >= 1 && (
                  <div className="mt-6 pt-6 border-t border-gray-200">
                    {/* Header with title and buttons in one row */}
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold text-gray-800">
                        Selected images: {images.length}
                      </h3>
                      <div className="flex items-center gap-3">
                        <button
                          onClick={handleClearAll}
                          className="px-4 py-2 text-sm font-medium text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors border border-red-200"
                        >
                          Clear All
                        </button>
                        <button
                          onClick={handleCalibrate}
                          disabled={loading}
                          className="px-5 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium text-sm flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
                        >
                          {loading ? (
                            <>
                              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                              </svg>
                              Calibrating...
                            </>
                          ) : (
                            <>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              Start Calibration
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                    
                    {/* Images preview section */}
                    <div className="mt-4">
                      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
                        {images.map(({ id, file }, index) => {
                          return (
                            <ImagePreviewItem
                              key={id}
                              file={file}
                              index={index}
                              onRemove={() => handleRemoveImage(id)}
                            />
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Camera Mode */}
          {(mode === 'camera' || mode === 'self-camera') && !results && (
            <div className="max-w-4xl mx-auto">
              <div className="bg-white rounded-xl shadow-lg p-8">
                <CameraCapture 
                  onImagesSelected={handleImagesSelected} 
                  targetCount={calibrationType === 'self' ? 50 : 15}
                  calibrationType={calibrationType}
                />
                
                {images.length >= (calibrationType === 'self' ? 10 : 5) && (
                  <div className="mt-6 flex justify-center">
                    <button
                      onClick={handleCalibrate}
                      disabled={loading}
                      className="px-8 py-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-semibold text-lg flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? (
                        <>
                          <svg className="animate-spin h-6 w-6" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Calibrating...
                        </>
                      ) : (
                        <>
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Start Calibration
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="max-w-4xl mx-auto mt-6">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <svg className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <h4 className="font-semibold text-red-800 mb-1">Calibration Error</h4>
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {results && <Results results={results} />}
        </main>

      </div>
    );
  }

  export default App;

