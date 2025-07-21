import { useState, type JSX } from "react";

const initialState = (contentMap: Map<string, JSX.Element>) => {
  const firstContentEntry = contentMap.entries().next().value!;
  return {
    name: firstContentEntry[0],
    component: firstContentEntry[1],
  };
};

export const useNavigationTabsState = (
  contentMap: Map<string, JSX.Element>
) => {
  const [activeContent, setActiveContent] = useState(() =>
    initialState(contentMap)
  );

  const updateActiveContent = (tabName: string) => {
    setActiveContent({
      name: tabName,
      component: contentMap.get(tabName)!,
    });
  };

  return {
    activeContent,
    updateActiveContent,
  };
};
