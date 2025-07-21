const typeToParameters = new Map([
  ["Erdős-Rényi", ["Number of Agents", "Probability"]],
  [
    "Watts-Strogatz",
    ["Number of Agents", "K-Neighbors", "Rewiring Probability"],
  ],
  ["Barabási-Albert", ["Number of Agents", "Number of Edges to Attach"]],
  ["Complete graph", ["Number of Agents"]],
]);

export const getParametersByType = (type: string) => {
  return typeToParameters.get(type);
};
