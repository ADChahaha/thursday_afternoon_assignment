"use client";

import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "../ui/label";
import { cn } from "@/lib/utils"; // 或你自己封装的 cn

interface Props {
  setImageFile: (file: File | null) => void; // 父组件回调
}

export default function ImageUploader({ setImageFile }: Props) {
  const [preview, setPreview] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageFile(file); // 父组件保存 File
    setPreview(URL.createObjectURL(file)); // 本地预览
  };

  const handleRemove = () => {
    setImageFile(null);
    setPreview(null);
  };

  return (
    <Card className="flex flex-col gap-2 p-4">
      {/* shadcn Input type="file" */}
      <Input
        key={preview}
        id="image-input"
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />

      {/* 预览 + 移除按钮 */}
      {preview && (
        <CardContent className="flex flex-col gap-2 items-center mt-2">
          <img src={preview} alt="preview" className="w-48 h-auto rounded" />
        </CardContent>
      )}
      <div
        className={cn("grid gap-2", preview ? "grid-cols-2" : "grid-cols-1")}
      >
        <Label htmlFor="image-input" className="block">
          <Button variant="default" asChild className="w-full" size="sm">
            <span>选择图片</span>
          </Button>
        </Label>
        {preview && (
          <Button onClick={handleRemove} variant="destructive" size="sm">
            移除图片
          </Button>
        )}
      </div>
    </Card>
  );
}
