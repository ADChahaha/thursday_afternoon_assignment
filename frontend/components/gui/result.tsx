"use client";

import React from "react";
import axios from "axios";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface Props {
  imageFile: File | null;
}

interface ApiResponse {
  type: string;
  cam: string; // base64
}

export default function Result({ imageFile }: Props) {
  const fetchResult = async (): Promise<ApiResponse | null> => {
    if (!imageFile) return null;

    const formData = new FormData();
    formData.append("file", imageFile);

    const res = await axios.post("/api/detect", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    return res.data;
  };

  const { data, refetch, isFetching } = useQuery({
    queryKey: ["process", imageFile],
    queryFn: fetchResult,
    enabled: false,
  });

  return (
    <Card className="flex flex-col gap-2 p-4">
      <Button
        className="select-none"
        onClick={() => refetch()}
        disabled={!imageFile || isFetching}
      >
        {isFetching ? "处理中..." : "开始处理"}
      </Button>

      {data && (
        <CardContent className="flex flex-col gap-2 mt-2 items-center">
          <p>类型：{data.type}</p>
          <img src={data.cam} alt="processed" className="w-48 h-auto rounded" />
        </CardContent>
      )}
    </Card>
  );
}
