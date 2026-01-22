# Fechamento do Modelo v1

## Objetivo
Detecção de fraudes bancárias em transações simuladas (BankSim).

## Modelo escolhido
Random Forest Classifier

## Justificativa
- Melhor equilíbrio entre recall e precisão
- Boa estabilidade
- Baixo risco de overfitting

## Limitações conhecidas
- Dados altamente desbalanceados
- Modelo treinado em dados sintéticos
- Possível degradação em novos períodos

## Próximos passos
- Monitorar drift
- Avaliar em dados v2
- Considerar tuning e novas features
