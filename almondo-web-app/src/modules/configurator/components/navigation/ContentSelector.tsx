import { useNavigationTabsState } from "../../hooks/useNavigationTabsState";
import type { JSX } from "react";

const ContentSelector = ({
  contentMap,
}: {
  contentMap: Map<string, JSX.Element>;
}) => {
  const { activeContent, updateActiveContent } =
    useNavigationTabsState(contentMap);
  const tabs: string[] = [];

  for (const tabName of contentMap.keys()) {
    tabs.push(tabName);
  }

  return (
    <div className="w-full">
      <ul className="flex flex-row justify-center mt-4">
        {tabs.map((name) => (
          <div
            key={name}
            className={`border-b px-2 ${
              name === activeContent.name
                ? "text-gray-900 border-gray-900"
                : "text-gray-500 border-gray-500"
            }`}
          >
            <li
              className="cursor-pointer"
              onClick={() => updateActiveContent(name)}
            >
              {name}
            </li>
          </div>
        ))}
      </ul>
      <div className="mt-8">{activeContent.component}</div>
    </div>
  );
};

export default ContentSelector;
