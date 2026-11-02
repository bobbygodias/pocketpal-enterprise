import {runInAction} from 'mobx';
import type {NativeBackendDeviceInfo} from 'llama.rn';

jest.mock('../ModelStore', () => ({
  modelStore: {
    context: {},
    engine: {},
    activeModel: {ggufMetadata: {n_layers: 32}},
    activeContextSettings: {
      n_gpu_layers: 8,
      devices: ['OpenCL'],
      flash_attn_type: 'off',
    },
    contextInitParams: {
      n_gpu_layers: 8,
      devices: ['OpenCL'],
      flash_attn_type: 'off',
    },
  },
}));

jest.mock('../../specs/NativeHardwareInfo', () => ({
  __esModule: true,
  default: {
    getGPUInfo: jest.fn().mockResolvedValue({renderer: ''}),
    getVulkanInfo: jest.fn().mockResolvedValue('{"available":false}'),
  },
}));

jest.mock('../../utils/deviceSelection', () => ({
  getAvailableDevices: jest.fn().mockResolvedValue([]),
}));

import {enterpriseRuntimeStore} from '../EnterpriseRuntimeStore';

type BackendSettings = {
  n_gpu_layers: number;
  devices?: string[];
  flash_attn_type: 'auto' | 'on' | 'off';
};

const modelStore = jest.requireMock('../ModelStore').modelStore as {
  context: object | undefined;
  engine: object | undefined;
  activeModel: {ggufMetadata?: {n_layers: number}} | undefined;
  activeContextSettings: BackendSettings | undefined;
  contextInitParams: BackendSettings;
};

const settings: BackendSettings = {
  n_gpu_layers: 8,
  devices: ['OpenCL'],
  flash_attn_type: 'off',
};

describe('EnterpriseRuntimeStore backend reporting', () => {
  beforeEach(() => {
    modelStore.context = {};
    modelStore.engine = {};
    modelStore.activeModel = {ggufMetadata: {n_layers: 32}};
    modelStore.activeContextSettings = {...settings};
    modelStore.contextInitParams = {...settings};

    runInAction(() => {
      enterpriseRuntimeStore.backendDevices = [
        {type: 'gpu', deviceName: 'OpenCL'} as NativeBackendDeviceInfo,
      ];
    });
  });

  it('does not claim actual GPU execution from an explicit GPU request', () => {
    modelStore.contextInitParams = {
      ...settings,
      n_gpu_layers: 99,
    };
    expect(enterpriseRuntimeStore.effectiveBackend).toBe('unverified');
    expect(enterpriseRuntimeStore.effectiveGpuLayers).toBeNull();
  });

  it('does not claim actual hybrid execution from requested layer count', () => {
    expect(enterpriseRuntimeStore.effectiveBackend).toBe('unverified');
  });

  it('reports CPU when the active context explicitly uses CPU', () => {
    modelStore.activeContextSettings = {
      ...settings,
      n_gpu_layers: 0,
      devices: ['CPU'],
    };
    expect(enterpriseRuntimeStore.effectiveBackend).toBe('cpu');
    expect(enterpriseRuntimeStore.effectiveGpuLayers).toBe(0);
  });

  it('reports idle without an active native context', () => {
    modelStore.context = undefined;
    expect(enterpriseRuntimeStore.effectiveBackend).toBe('idle');
  });

  it('requires a reload when only flash attention changes', () => {
    modelStore.contextInitParams = {
      ...settings,
      flash_attn_type: 'auto',
    };
    expect(enterpriseRuntimeStore.requiresModelReload).toBe(true);
  });

  it('does not require a reload when all backend settings match', () => {
    expect(enterpriseRuntimeStore.requiresModelReload).toBe(false);
  });

  it('requires a reload when the device changes', () => {
    modelStore.contextInitParams = {
      ...settings,
      devices: ['CPU'],
    };
    expect(enterpriseRuntimeStore.requiresModelReload).toBe(true);
  });
});
