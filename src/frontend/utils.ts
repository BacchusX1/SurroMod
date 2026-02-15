import { PaletteItem, DataKind } from './types';

// ─── Palette definitions ────────────────────────────────────────────────────
// One entry per draggable building block shown in the sidebar.

const dataKinds: DataKind[] = ['scalar', '2d_field', '3d_field', 'time_series', 'step'];

/** Build input palette items – one per data kind (mirrors data_digester/) */
const inputItems: PaletteItem[] = dataKinds.map((dk) => ({
  category: 'input' as const,
  label: `Input (${dk.replace('_', ' ')})`,
  defaultData: {
    label: `Input (${dk.replace('_', ' ')})`,
    category: 'input' as const,
    dataKind: dk,
    source: '',
  },
}));

/** Regressor palette items – one per data kind (mirrors predictors/regressors/) */
const regressorItems: PaletteItem[] = dataKinds
  .filter((dk) => dk !== 'step') // no step regressor in backend
  .map((dk) => ({
    category: 'regressor' as const,
    label: `Regressor (${dk.replace('_', ' ')})`,
    defaultData: {
      label: `Regressor (${dk.replace('_', ' ')})`,
      category: 'regressor' as const,
      dataKind: dk,
      method: 'default',
    },
  }));

/** Classifier palette items (mirrors predictors/classifiers/) */
const classifierItems: PaletteItem[] = dataKinds
  .filter((dk) => dk !== 'step')
  .map((dk) => ({
    category: 'classifier' as const,
    label: `Classifier (${dk.replace('_', ' ')})`,
    defaultData: {
      label: `Classifier (${dk.replace('_', ' ')})`,
      category: 'classifier' as const,
      dataKind: dk,
      method: 'default',
    },
  }));

/** Validator palette items (mirrors analyzers/) */
const validatorItems: PaletteItem[] = [
  {
    category: 'validator',
    label: 'Classifier Validator',
    defaultData: {
      label: 'Classifier Validator',
      category: 'validator',
      validatorKind: 'classifier_validator',
    },
  },
  {
    category: 'validator',
    label: 'Regressor Validator',
    defaultData: {
      label: 'Regressor Validator',
      category: 'validator',
      validatorKind: 'regressor_validator',
    },
  },
  {
    category: 'validator',
    label: 'Relation Seeker',
    defaultData: {
      label: 'Relation Seeker',
      category: 'validator',
      validatorKind: 'relation_seeker',
    },
  },
];

export const paletteItems: PaletteItem[] = [
  ...inputItems,
  ...regressorItems,
  ...classifierItems,
  ...validatorItems,
];

/** Generate a unique id for a new node */
let _counter = 0;
export function nextNodeId(): string {
  return `node_${Date.now()}_${_counter++}`;
}

/** Category → colour mapping used across nodes */
export const categoryColor: Record<string, string> = {
  input: '#4ade80',      // green
  regressor: '#60a5fa',  // blue
  classifier: '#f472b6', // pink
  validator: '#facc15',  // yellow
};
