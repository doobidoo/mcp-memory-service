import { Config } from "@remotion/cli/config";

// Video output settings
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
Config.setCodec("h264");

// Performance settings
Config.setConcurrency(4);

// Output quality
// CRF: 18-23 recommended (lower = better quality, larger file)
// Default is 18 for high quality
