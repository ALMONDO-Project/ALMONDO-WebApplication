import { useState } from "react";

const CustomGraphForm = () => {
  const [format, setFormat] = useState("edgelist");

  const handleFormatSelection = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFormat(e.target.value);
  };
  return (
    <>
      <label htmlFor="upload-format" className="block mb-2 text-lg font-medium">
        Upload Format
      </label>
      <select
        onChange={(e) => handleFormatSelection(e)}
        id="upload-format"
        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5"
      >
        <option value="edgelist">Edge List</option>
        <option value="matrix">Adjacency matrix</option>
      </select>
      <label className="block mt-4 mb-2 text-lg font-medium">
        Upload {format === "edgelist" ? "Edge List" : "Adjacency Matrix"}
      </label>
      <input
        type="file"
        className="block w-full border border-gray-200 shadow-sm rounded-lg text-sm file:bg-gray-50 
            file:border-0
            file:me-4
            file:py-3 file:px-4
            cursor-pointer"
      />
    </>
  );
};

export default CustomGraphForm;
