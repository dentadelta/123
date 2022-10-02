import { useState, useEffect } from "react";
import axios from "axios";

export const ProgressBar = ({progress}) => {
    return (
        <div>
            <div className="progressbar">
                <div style={{
                    height: "100%",
                    width: `${progress}%`,
                    backgroundColor: "white",
                    transition: "width 1s ease-in-out",
                    zIndex: "3"

                }}></div>
                <span> {progress}%</span>
            </div>
        </div>
    )

}