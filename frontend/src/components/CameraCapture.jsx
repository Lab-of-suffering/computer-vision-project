import React, { useState, useRef, useEffect } from 'react';

const CameraCapture = ({
  onImagesSelected,
  targetCount = 15,
  showPreview = true,
  calibrationType = 'zhang'
}) => {
  const [stream, setStream] = useState(null);
  const [capturedImages, setCapturedImages] = useState([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    return () => {
      // Cleanup: stop camera stream when component unmounts
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      
      setStream(mediaStream);
      setIsCameraActive(true);
      
      // Use setTimeout to ensure DOM is ready
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          
          // Set up event handlers
          const handleLoadedMetadata = async () => {
            try {
              await videoRef.current.play();
              console.log('Video playing');
              setIsVideoReady(true);
            } catch (err) {
              console.error('Play error:', err);
              // Try one more time
              setTimeout(() => {
                videoRef.current.play().then(() => {
                  setIsVideoReady(true);
                }).catch(e => console.error('Retry failed:', e));
              }, 100);
            }
          };

          videoRef.current.addEventListener('loadedmetadata', handleLoadedMetadata);
          
          // Fallback: set ready after a delay if events don't fire
          setTimeout(() => {
            if (!isVideoReady && videoRef.current && videoRef.current.readyState >= 2) {
              console.log('Fallback: setting video ready');
              setIsVideoReady(true);
            }
          }, 2000);
        }
      }, 100);
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Failed to access camera. Please check permissions.');
      setIsCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setIsCameraActive(false);
    setIsVideoReady(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) {
      alert('Camera components not ready.');
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Check if video is ready
    if (!isVideoReady || video.readyState < 2) {
      alert('Video is not ready yet. Please wait a moment.');
      console.log('Video readyState:', video.readyState, 'isVideoReady:', isVideoReady);
      return;
    }

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    if (canvas.width === 0 || canvas.height === 0) {
      alert('Video dimensions are not available. Please try again.');
      console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
      return;
    }
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) {
        console.error('Failed to capture image from canvas');
        return;
      }
      
      const file = new File([blob], `capture_${Date.now()}.png`, { type: 'image/png' });
      const url = URL.createObjectURL(blob);
      
      const newImages = [...capturedImages, { file, url }];
      setCapturedImages(newImages);
      onImagesSelected(newImages.map(img => img.file));

      if (newImages.length >= targetCount) {
        alert(`You've captured ${targetCount} images! Ready for calibration.`);
      }
    }, 'image/png');
  };

  const removeImage = (index) => {
    const newImages = capturedImages.filter((_, i) => i !== index);
    setCapturedImages(newImages);
    onImagesSelected(newImages.map(img => img.file));
  };

  const clearAll = () => {
    setCapturedImages([]);
    onImagesSelected([]);
  };

  const imageTypeLabel = calibrationType === 'self' ? 'scene' : 'chessboard';

  return (
    <div className="w-full">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">
            Capture {targetCount} {imageTypeLabel} images
          </h3>
          <div className="flex gap-2">
            {!isCameraActive ? (
              <button
                onClick={startCamera}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
              >
                Stop Camera
              </button>
            )}
          </div>
        </div>

        {isCameraActive && (
          <div className="relative bg-black rounded-lg overflow-hidden" style={{ minHeight: '400px' }}>
            {!isVideoReady && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                  <p>Loading camera...</p>
                </div>
              </div>
            )}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-auto"
              style={{ display: isVideoReady ? 'block' : 'none' }}
            />
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-4">
              <button
                onClick={captureImage}
                className="px-6 py-3 bg-white text-gray-900 rounded-full hover:bg-gray-100 transition-colors font-semibold flex items-center gap-2 shadow-lg"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Capture ({capturedImages.length}/{targetCount})
              </button>
            </div>
            
            {capturedImages.length > 0 && capturedImages.length < targetCount && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-yellow-500 text-white px-4 py-2 rounded-full font-semibold">
                Remaining: {targetCount - capturedImages.length}
              </div>
            )}
            
            {capturedImages.length >= targetCount && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-4 py-2 rounded-full font-semibold animate-pulse">
                ✓ Ready! You can calibrate now
              </div>
            )}
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />
      </div>

      {showPreview && capturedImages.length > 0 && (
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">
              Captured: {capturedImages.length} of {targetCount}
            </h3>
            <button
              onClick={clearAll}
              className="px-4 py-2 text-sm font-medium text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors"
            >
              Delete all
            </button>
          </div>
          
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
            {capturedImages.map((image, index) => (
              <div key={index} className="relative group">
                <img
                  src={image.url}
                  alt={`Capture ${index + 1}`}
                  className="w-full h-24 object-cover rounded-lg border-2 border-gray-200"
                />
                <button
                  onClick={() => removeImage(index)}
                  className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                <div className="absolute bottom-1 left-1 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                  {index + 1}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {!showPreview && capturedImages.length > 0 && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2 text-green-700">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium">Captured: {capturedImages.length} of {targetCount}</span>
          </div>
        </div>
      )}

      {!isCameraActive && capturedImages.length === 0 && (
        <div className="text-center py-12 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <p className="text-gray-500 mb-2">Click "Start Camera" to begin</p>
          <p className="text-sm text-gray-400">Hold the chessboard at different angles</p>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;

