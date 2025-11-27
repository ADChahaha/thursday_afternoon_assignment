"use client";

import React, { useState } from "react";
import ImageUploader from "./imageuploader";
import Result from "./result";

export default function Feed() {
  // 父组件保存 File
  const [imageFile, setImageFile] = useState<File | null>(null);

  return (
    <div className="p-4 grid grid-cols-2 gap-4">
      {/* 上传组件只负责选择图片，把 File 回调给父组件 */}
      <ImageUploader setImageFile={setImageFile} />

      {/* Result 组件接收父组件传来的 File */}
      <Result imageFile={imageFile} />
    </div>
  );
}
