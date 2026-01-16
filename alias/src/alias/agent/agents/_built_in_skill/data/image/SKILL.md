---
name: image-file
description: Guildlines for handling image files
type: image
---

# Images Handling Specifications

# Inspection
- Use PIL to get image dimensions and format
- Invoke vision tools to extract content from images when needed
- Always inspect image properties before processing

# Best Practices
- Validate image formats before processing
- Handle corrupted or unsupported image files gracefully
- Optimize image processing operations for efficiency
- Consider using appropriate libraries/tools for specific image processing tasks
- Process images individually or in small batches to prevent system crashes
- Consider memory usage when working with large or high-resolution images
