import { useState } from "react";
import Step from "../model/Step";
import { getConfiguratorSteps } from "../logic/configuratorSteps";

export const useSteps = () => {
  const [steps, setSteps] = useState<Step[]>(getConfiguratorSteps());
  const [currentStep, setCurrentStep] = useState<number>(0);

  const navigateNext = () => {
    setSteps((prevSteps) =>
      prevSteps.map((step) => {
        if (step.number === currentStep) {
          step.complete();
        }
        return step;
      })
    );
    setCurrentStep((currentStep) => currentStep + 1);
  };

  const navigatePrevious = () => {
    setCurrentStep((currentStep) => currentStep - 1);
  };

  const setCurrentStepCompletable = () => {
    setSteps((prevSteps) =>
      prevSteps.map((step) => {
        if (step.number === currentStep) {
          step.makeCompletable();
        }
        return step;
      })
    );
  };

  return {
    steps,
    currentStep,
    navigateNext,
    navigatePrevious,
    setCurrentStepCompletable,
  };
};
