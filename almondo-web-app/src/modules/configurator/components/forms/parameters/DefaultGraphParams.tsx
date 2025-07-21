import { getParametersByType } from "../../../logic/graphParamsEvaluator";

const DefaultGraphParams = ({ graphType }: { graphType: string }) => {
  return (
    <>
      {getParametersByType(graphType)!.map((label) => (
        <div key={label} className="mb-5">
          <label className="block mb-2 text-sm font-medium text-gray-900">
            {label}
          </label>
          <input className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5" />
        </div>
      ))}
    </>
  );
};

export default DefaultGraphParams;
