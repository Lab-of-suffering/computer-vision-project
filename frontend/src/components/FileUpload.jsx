import React, { useState, useRef, useEffect } from 'react';

const FileUpload = ({ onImagesSelected, currentImages = [], minImages = 5, maxImages = 500, label = "Upload images", showPreview = true }) => {
  const [previewUrls, setPreviewUrls] = useState([]);
  const fileInputRef = useRef(null);

  // Update preview URLs only when currentImages changes AND showPreview is true
  useEffect(() => {
    if (showPreview) {
      if (currentImages.length > 0) {
        const urls = currentImages.map(file => URL.createObjectURL(file));
        setPreviewUrls(prevUrls => {
          // Revoke old URLs before setting new ones
          prevUrls.forEach(url => URL.revokeObjectURL(url));
          return urls;
        });
      } else {
        setPreviewUrls(prevUrls => {
          prevUrls.forEach(url => URL.revokeObjectURL(url));
          return [];
        });
      }
      
      // Cleanup function to revoke URLs when component unmounts
      return () => {
        setPreviewUrls(prevUrls => {
          prevUrls.forEach(url => URL.revokeObjectURL(url));
          return [];
        });
      };
    }
  }, [currentImages, showPreview]);

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    
    if (newFiles.length === 0) {
      return;
    }
    
    // Filter out duplicates (check if file with same name, size and date already exists)
    const uniqueNewFiles = newFiles.filter(newFile => {
      return !currentImages.some(existingFile => 
        existingFile.name === newFile.name && 
        existingFile.size === newFile.size &&
        existingFile.lastModified === newFile.lastModified
      );
    });

    if (uniqueNewFiles.length === 0) {
      // All files are duplicates
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }
    
    // Add new unique files to existing ones passed via props
    const allFiles = [...currentImages, ...uniqueNewFiles];
    
    // Check minimum only for total count
    if (allFiles.length < minImages) {
      alert(`Please upload at least ${minImages} images in total`);
      // Still add the files, just warn
    }

    // Just notify parent about new complete list
    onImagesSelected(allFiles);
    
    // Reset input to allow selecting the same files again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = (index) => {
    // Filter based on index from the prop-based list
    const newFiles = currentImages.filter((_, i) => i !== index);
    onImagesSelected(newFiles);
  };

  const handleClearAll = () => {
    onImagesSelected([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      <div className="mb-6">
        <label className="block mb-2 text-sm font-medium text-gray-700">
          {label}
        </label>
        <div className="flex items-center justify-center w-full">
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg
                className="w-12 h-12 mb-4 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="mb-2 text-sm text-gray-500">
                <span className="font-semibold">Click to select</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">
                PNG, JPG (minimum {minImages} images total)
              </p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {showPreview && currentImages.length > 0 && (
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">
              Selected images: {currentImages.length}
            </h3>
            <button
              onClick={handleClearAll}
              className="px-4 py-2 text-sm font-medium text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors"
            >
              Clear all
            </button>
          </div>
          
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
            {previewUrls.map((url, index) => (
              <div key={index} className="relative group">
                <img
                  src={url}
                  alt={`Preview ${index + 1}`}
                  className="w-full h-24 object-cover rounded-lg border-2 border-gray-200"
                />
                <button
                  onClick={() => handleRemoveFile(index)}
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
    </div>
  );
};

export default FileUpload;
