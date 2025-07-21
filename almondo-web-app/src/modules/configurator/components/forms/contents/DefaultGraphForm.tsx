import { useState } from "react";
import DefaultGraphParams from "../parameters/DefaultGraphParams";

const DefaultGraphForm = () => {
  const [graphType, setGraphType] = useState("Erdős-Rényi");

  const handleGraphTypeSelection = (
    e: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setGraphType(e.target.value);
  };

  return (
    <>
      <label htmlFor="graph-types" className="block mb-2 text-lg font-medium">
        Graph Type
      </label>
      <select
        onChange={(e) => handleGraphTypeSelection(e)}
        id="graph-types"
        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5"
      >
        <option value="Erdős-Rényi">Erdős-Rényi</option>
        <option value="Watts-Strogatz">Watts-Strogatz</option>
        <option value="Barabási-Albert">Barabási-Albert</option>
        <option value="Complete graph">Complete graph</option>
      </select>
      <div className="py-3 flex items-center text-base before:flex-1 before:border-t before:border-gray-200 before:me-6 after:flex-1 after:border-t after:border-gray-200 after:ms-6 dark:text-white dark:before:border-neutral-600 dark:after:border-neutral-600">
        Parameters
      </div>
      <DefaultGraphParams graphType={graphType} />
    </>
  );
};

export default DefaultGraphForm;
